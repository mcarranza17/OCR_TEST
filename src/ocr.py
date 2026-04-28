from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from src.image_io import image_bytes_to_bgr


class OCRError(RuntimeError):
    pass


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "confidence": self.confidence}


@dataclass(frozen=True)
class OCRResult:
    full_text: str
    average_confidence: float
    lines: list[OCRLine]


def get_paddle_ocr(lang: str = "es"):
    """Construct a PaddleOCR instance. Expensive — should be cached upstream."""
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str((Path(".models") / "paddlex").resolve()))

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRError("PaddleOCR no está instalado. Ejecuta pip install -r requirements.txt") from exc

    # paddleocr 3.x uses `use_textline_orientation`; 2.x uses `use_angle_cls`.
    try:
        return PaddleOCR(use_textline_orientation=True, lang=lang)
    except TypeError:
        return PaddleOCR(use_angle_cls=True, lang=lang)


def run_ocr_with(ocr_instance, image_bytes: bytes) -> OCRResult:
    image = image_bytes_to_bgr(image_bytes)

    def _call(img):
        # paddleocr 3.x exposes `.predict()`; 2.x only has `.ocr()`.
        if hasattr(ocr_instance, "predict"):
            return ocr_instance.predict(img)
        return ocr_instance.ocr(img)

    # Paddle's MKL-DNN sometimes throws "could not execute a primitive" on
    # specific tensor shapes. We try the original and a few rescaled copies
    # silently so the user never sees the failure.
    import cv2

    h, w = image.shape[:2]
    candidates = [
        image,
        cv2.resize(image, (max(2, int(w * 0.95)), max(2, int(h * 0.95)))),
        cv2.resize(image, (max(2, int(w * 1.05)), max(2, int(h * 1.05)))),
        cv2.resize(image, (max(2, int(w * 0.85)), max(2, int(h * 0.85)))),
    ]

    raw_result = None
    for candidate in candidates:
        try:
            raw_result = _call(candidate)
            break
        except RuntimeError:
            continue

    if raw_result is None:
        raise OCRError("") from None

    lines = _extract_lines(raw_result)
    if not lines:
        raise OCRError("No se detectó texto en el documento.")

    full_text = "\n".join(line.text for line in lines)
    avg_conf = sum(line.confidence for line in lines) / len(lines)
    return OCRResult(full_text=full_text, average_confidence=avg_conf, lines=lines)


def run_ocr(image_bytes: bytes, lang: str = "es") -> OCRResult:
    return run_ocr_with(get_paddle_ocr(lang), image_bytes)


def _extract_lines(raw_result) -> list[OCRLine]:
    lines: list[OCRLine] = []
    for page in raw_result or []:
        if isinstance(page, dict) and "rec_texts" in page:
            texts = page.get("rec_texts") or []
            scores = page.get("rec_scores") or []
            for index, text in enumerate(texts):
                confidence = scores[index] if index < len(scores) else 0.0
                if text:
                    lines.append(OCRLine(text=str(text).strip(), confidence=float(confidence)))
            continue

        for item in page or []:
            if len(item) < 2:
                continue
            text, confidence = item[1]
            if text:
                lines.append(OCRLine(text=str(text).strip(), confidence=float(confidence)))

    return lines
