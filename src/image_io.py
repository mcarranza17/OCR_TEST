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
    try:
        import cv2
    except ImportError as exc:
        raise ValueError("opencv-python no esta instalado. Ejecuta pip install -r requirements.txt") from exc

    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is not None:
        return image

    # Fallback to Pillow (supports HEIC/HEIF when pillow-heif is installed,
    # plus exotic formats OpenCV may not handle).
    _register_heif_once()
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise ValueError(
            "No se pudo decodificar la imagen. Instala Pillow y pillow-heif "
            "(pip install -r requirements.txt) o usa JPG/PNG."
        ) from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = ImageOps.exif_transpose(img)
            rgb = img.convert("RGB")
            arr = np.asarray(rgb, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001 - any decode failure surfaces as ValueError.
        raise ValueError(
            "No se pudo decodificar la imagen. Formato no soportado o archivo corrupto."
        ) from exc

    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
