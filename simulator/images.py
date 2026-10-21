"""Load image resources and crop Feed/IPV viewports."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from http.client import HTTPSConnection, HTTPException
from ipaddress import ip_address
from io import BytesIO
from pathlib import Path, PureWindowsPath
import socket
from typing import Callable, Optional, Tuple
from urllib.parse import urlsplit

from . import geometry


HTTP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

DEFAULT_IMAGE_HOSTS = frozenset({"img.alicdn.com"})


def _image_url(source, allowed_hosts):
    """Validate remote references before any cache, DNS, or network access."""
    url = urlsplit(source)
    if not url.scheme and not url.netloc:
        return None  # Local references receive a separate root-containment check.
    if (url.scheme != "https"
            or any(ord(char) <= 32 or ord(char) == 127 for char in source)
            or url.username is not None or url.password is not None
            or url.hostname not in allowed_hosts
            or url.port not in (None, 443)
            or url.netloc.lower() not in (url.hostname, f"{url.hostname}:443")):
        raise ValueError("Image URL must use an allowed HTTPS host on port 443")
    return url


def _public_address(address):
    if "%" in address:
        return False
    ip = ip_address(address)
    if (not ip.is_global or ip.is_multicast or ip.is_reserved
            or ip.is_loopback or ip.is_link_local or ip.is_unspecified):
        return False
    if ip.version == 6 and (ip.ipv4_mapped is not None or ip.is_site_local):
        return False
    return True


class _PublicHTTPSConnection(HTTPSConnection):
    """Connect to validated numeric addresses while retaining TLS hostname checks."""

    def connect(self):
        addresses = socket.getaddrinfo(
            self.host, self.port, type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP,
        )
        if not addresses:
            raise OSError("Image host has no addresses")
        # Reject mixed public/private answers before opening any socket.
        if any(family not in (socket.AF_INET, socket.AF_INET6)
               or not _public_address(sockaddr[0])
               or (family == socket.AF_INET6 and sockaddr[3] != 0)
               for family, _, _, _, sockaddr in addresses):
            raise ValueError("Image host resolves to a non-public address")
        last_error = None
        for family, kind, protocol, _, sockaddr in addresses:
            sock = socket.socket(family, kind, protocol)
            try:
                sock.settimeout(self.timeout)
                # Use the checked sockaddr directly: no second DNS lookup.
                sock.connect(sockaddr)
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                return
            except OSError as exc:
                sock.close()
                last_error = exc
        raise last_error


def _read_remote_image(url, timeout):
    connection = _PublicHTTPSConnection(url.hostname, port=443, timeout=timeout)
    try:
        target = url.path or "/"
        if url.query:
            target += "?" + url.query
        connection.request("GET", target, headers={"User-Agent": HTTP_USER_AGENT})
        with connection.getresponse() as response:
            if 300 <= response.status < 400:
                raise ValueError("Image redirects are disabled")
            if response.status != 200:
                raise OSError(f"Image server returned HTTP {response.status}")
            return response.read()
    finally:
        connection.close()

ImageRef = str | tuple[str, ...]


def image_ref(value) -> ImageRef:
    return tuple(value) if isinstance(value, (list, tuple)) else value or ""


class MissingAssetError(RuntimeError):
    """A required resource could not be loaded; no substitute image was used."""


class ImageStore:
    """Load local or allowlisted HTTPS images with a bounded LRU cache.

    ``allowed_hosts``, ``sources``, and ``fetch`` are trusted caller configuration,
    never dataset fields. URL and path restrictions apply to the built-in loader.
    A custom ``fetch`` receives the mapped source and owns its I/O security.
    """

    def __init__(self, root=".", *, sources=None, fetch: Callable | None = None,
                 cache_dir=None, max_cached_images=8, timeout=30,
                 allowed_hosts=DEFAULT_IMAGE_HOSTS):
        try:
            self.root = Path(root).resolve()
        except (OSError, RuntimeError) as exc:
            raise MissingAssetError(
                f"Cannot resolve image root ({type(exc).__name__})"
            ) from None
        self.sources = sources or {}
        self.fetch = fetch
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_cached_images = max_cached_images
        self.timeout = timeout
        self.allowed_hosts = frozenset(host.lower() for host in allowed_hosts)
        self._cache = OrderedDict()
        self._sizes = {}

    def _local_path(self, source):
        path = Path(source)
        windows_path = PureWindowsPath(source)
        if path.is_absolute() or windows_path.drive or windows_path.root:
            raise ValueError("Local images must use paths relative to the image root")
        try:
            resolved = (self.root / path).resolve()
        except RuntimeError:
            raise ValueError("Local image path cannot be resolved") from None
        if not resolved.is_relative_to(self.root):
            raise ValueError("Local image path escapes the image root")
        return resolved

    def load(self, reference: str):
        from PIL import Image

        source = self.sources.get(reference, reference)
        if not isinstance(source, str) or not source:
            raise MissingAssetError("Image source is missing")
        key = sha256(source.encode()).hexdigest()
        try:
            if self.fetch is None:
                # Validate the built-in loader's mapped source before either cache.
                url = _image_url(source, self.allowed_hosts)
                path = self._local_path(source) if url is None else None
            if source in self._cache:
                self._cache.move_to_end(source)
                self._sizes[reference] = self._cache[source].size
                return self._cache[source]
            if self.fetch is not None:
                image = self.fetch(source).convert("RGB")
            else:
                cached = self.cache_dir / (key + ".image") if self.cache_dir else None
                if cached and cached.is_file():
                    payload = cached.read_bytes()
                elif url is not None:
                    payload = _read_remote_image(url, self.timeout)
                else:
                    payload = path.read_bytes()
                with Image.open(BytesIO(payload)) as opened:
                    image = opened.convert("RGB")
                if cached and not cached.exists():
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(payload)
        except (OSError, ValueError, HTTPException) as exc:
            # Underlying file/cache exceptions can contain absolute local paths.
            raise MissingAssetError(
                f"Cannot load image asset {key[:12]} ({type(exc).__name__})"
            ) from None
        self._cache[source] = image
        self._sizes[reference] = image.size
        while len(self._cache) > self.max_cached_images:
            self._cache.popitem(last=False)
        return image

    def size(self, reference):
        if reference not in self._sizes:
            self.load(reference)
        return self._sizes[reference]

    def document(self, reference, *, meta=None):
        return VerticalImage(self, reference, meta=meta)


class VerticalImage:
    """Crop contiguous, unscaled image parts in top-to-bottom order."""

    def __init__(self, store: ImageStore, reference, *, meta=None):
        normalized = image_ref(reference)
        if not normalized:
            raise MissingAssetError("Long image reference is missing")
        self.store = store
        self.parts = (normalized,) if isinstance(normalized, str) else normalized
        meta = meta or {}
        heights = meta.get("part_heights")
        width = meta.get("image_width")
        if heights is not None:
            self._heights = tuple(heights)
            self.width = width
        else:
            sizes = [store.size(part) for part in self.parts]
            self.width = sizes[0][0]
            self._heights = tuple(h for w, h in sizes)
        self.height = sum(self._heights)
        self.size = (self.width, self.height)

    def crop(self, box):
        from PIL import Image

        left, top, right, bottom = (int(v) for v in box)
        result = Image.new("RGB", (right - left, bottom - top))
        offset = 0
        for reference, height in zip(self.parts, self._heights):
            start, end = max(top, offset), min(bottom, offset + height)
            if start < end:
                part = self.store.load(reference)
                piece = part.crop((left, start - offset, right, end - offset))
                result.paste(piece, (0, start - top))
            offset += height
        return result


STATUS_BAR_HEIGHT = geometry.SCREEN_HEIGHT - geometry.IPV_VIEWPORT_HEIGHT

PURCHASE_BAR_HEIGHT = 55


@dataclass(frozen=True)
class Capture:
    """One product's full-page detail capture, as the log describes it."""

    url: ImageRef
    image_width: int
    image_height: int

    @classmethod
    def from_meta(cls, url: ImageRef, meta: Optional[dict]) -> Optional["Capture"]:
        """Build from prepared catalog image metadata."""
        return cls(image_ref(url), meta["image_width"], meta["image_height"])

    @property
    def content_height(self) -> int:
        """Scrollable height: the document minus the status bar above it."""
        return max(0, self.image_height - STATUS_BAR_HEIGHT)

    @property
    def max_scroll_y(self) -> int:
        """Deepest offset a shopper can reach, in content pixels."""
        return max(0, self.content_height - geometry.IPV_VIEWPORT_HEIGHT)

    def page_layout(self) -> geometry.PageLayout:
        """Geometry of this page with no block positions."""
        return geometry.PageLayout(blocks=(), total_height=self.content_height)


def crop_box(capture: Capture, scroll_y: int) -> Tuple[int, int, int, int]:
    """``(left, top, right, bottom)`` of the content rows visible at ``scroll_y``."""
    offset = min(max(0, int(scroll_y)), capture.max_scroll_y)
    top = STATUS_BAR_HEIGHT + offset
    return (0, top, capture.image_width, top + geometry.IPV_VIEWPORT_HEIGHT)

def composite_screen(image, capture: Capture, scroll_y: int):
    """One device screen at ``scroll_y``, built from the captured document."""
    from PIL import Image

    width = capture.image_width
    screen = Image.new("RGB", (width, geometry.SCREEN_HEIGHT))
    screen.paste(image.crop((0, 0, width, STATUS_BAR_HEIGHT)), (0, 0))
    screen.paste(image.crop(crop_box(capture, scroll_y)), (0, STATUS_BAR_HEIGHT))
    bar_top = capture.image_height - PURCHASE_BAR_HEIGHT
    screen.paste(
        image.crop((0, bar_top, width, capture.image_height)),
        (0, geometry.SCREEN_HEIGHT - PURCHASE_BAR_HEIGHT),
    )
    return screen


_TOP = round(geometry.CHROME_TOP)
_BOTTOM = round(geometry.CHROME_BOTTOM)
_WIDTH = geometry.SCREEN_WIDTH
_HEIGHT = geometry.SCREEN_HEIGHT
_CONTENT = _HEIGHT - _TOP - _BOTTOM


class ImageRenderer:
    def __init__(self, store=None):
        self.store = store or ImageStore()
        self._documents = {}

    def bind_session(self, raw):
        from .data import feed_catalog_items

        self._documents = {}
        self._feed_ref = image_ref(raw["feed_long_image"]["url"])
        self._views = {}
        self._capture_meta = {}
        for item in feed_catalog_items(raw):
            biz = str(item.get("item_id") or "")
            if not biz:
                continue
            views = self._views.setdefault(biz, {})
            for view in ("first", "long", "params", "reviews"):
                ref = image_ref(item.get(f"ipv_{view}_image"))
                if ref:
                    views.setdefault(view, ref)
            if item.get("ipv_capture_meta"):
                self._capture_meta.setdefault(biz, item["ipv_capture_meta"])

    def _document(self, ref, meta=None):
        ref = image_ref(ref)
        if not ref:
            raise MissingAssetError("Required page image is missing")
        if ref not in self._documents:
            self._documents[ref] = self.store.document(ref, meta=meta)
        return self._documents[ref]

    def _screen(self, ref):
        document = self._document(ref)
        return document.crop((0, 0, _WIDTH, _HEIGHT))

    def render(self, observation):
        if observation.screen_type == "feed":
            return self._feed_screen(observation.scroll_position)
        if observation.screen_type == "ipv":
            return self._product_screen(observation.snapshot)
        return None

    def _feed_screen(self, scroll_position):
        from PIL import Image

        document = self._document(self._feed_ref)
        offset = min(max(0, int(scroll_position)), max(0, document.height - _CONTENT))
        screen = Image.new("RGB", (_WIDTH, _HEIGHT), "white")
        screen.paste(document.crop((0, offset, _WIDTH, offset + _CONTENT)), (0, _TOP))
        return screen

    def page_layout(self, item_id):
        biz = str(item_id)
        ref = self._views.get(biz, {}).get("long")
        document = self._document(ref, self._capture_meta.get(biz))
        return Capture(image_ref(ref), document.width, document.height).page_layout()

    def _product_screen(self, snapshot):
        from . import ipv_snapshot

        if snapshot is None:
            return None
        biz = str(snapshot.item_id)
        views = self._views.get(biz, {})
        if snapshot.params_open:
            return self._screen(views.get("params"))
        if snapshot.subpage == ipv_snapshot.SUBPAGE_COMMENT_LIST:
            return self._screen(views.get("reviews"))
        if snapshot.subpage != ipv_snapshot.SUBPAGE_MAIN or snapshot.sku_open:
            return None
        if int(snapshot.scroll_y or 0) <= 0:
            return self._screen(views.get("first"))
        ref = views.get("long")
        document = self._document(ref, self._capture_meta.get(biz))
        capture = Capture(image_ref(ref), document.width, document.height)
        return composite_screen(document, capture, int(snapshot.scroll_y))

    @staticmethod
    def observation_note(observation):
        from . import ipv_snapshot

        page = observation.snapshot
        if observation.screen_type != "ipv" or page is None or page.params_open:
            return ""
        if page.subpage == ipv_snapshot.SUBPAGE_MAIN and not page.sku_open:
            if int(page.hero_index or 0) != 0:
                return "Image note: after swiping, the product's first screenshot is a reference; it does not show the actual carousel index."
        return ""
