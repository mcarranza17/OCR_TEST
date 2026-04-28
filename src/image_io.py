from __future__ import annotations

import io

import numpy as np


_HEIF_REGISTERED = False


def _register_heif_once() -> None:
    global _HEIF_REGISTERED
    if _HEIF_REGISTERED:
        return
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except Exception:  # noqa: BLE001 - missing optional dep is non-fatal.
        pass
    _HEIF_REGISTERED = True


def image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decode any supported image format to a BGR numpy array.

    Always applies EXIF orientation, so iPhone photos saved sideways with a
    rotation tag end up upright. Falls back to OpenCV only if Pillow is not
    available.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("opencv-python no está instalado. Ejecuta pip install -r requirements.txt") from exc

    _register_heif_once()
    try:
        from PIL import Image, ImageOps
    except ImportError:
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "No se pudo decodificar la imagen. Instala Pillow y pillow-heif "
                "(pip install -r requirements.txt) o usa JPG/PNG."
            )
        return image

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
    except Exception:
        # Last resort: cv2 without EXIF awareness.
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(
                "No se pudo decodificar la imagen. Formato no soportado o archivo corrupto."
            )
        return image

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def normalize_upright_jpeg(image_bytes: bytes, quality: int = 92) -> bytes:
    """Re-encode any input as upright JPEG with EXIF rotation already applied.

    Use this on uploaded files before storing or previewing, so the bytes
    saved on disk and shown in `st.image` are consistent with what the
    pipeline (`image_bytes_to_bgr`) sees.
    """
    _register_heif_once()
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return image_bytes

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
            buf = io.BytesIO()
            rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()
    except Exception:  # noqa: BLE001 - if decode fails, return original bytes.
        return image_bytes
