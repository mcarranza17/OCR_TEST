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


def run_ocr(image_bytes: bytes, lang: str = "es") -> OCRResult:
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str((Path(".models") / "paddlex").resolve()))

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRError("PaddleOCR no esta instalado. Ejecuta pip install -r requirements.txt") from exc

    image = image_bytes_to_bgr(image_bytes)
    ocr = PaddleOCR(use_textline_orientation=True, lang=lang)
    raw_result = ocr.predict(image)

    lines = _extract_lines(raw_result)
    if not lines:
        raise OCRError("No se detecto texto en el documento.")

    full_text = "\n".join(line.text for line in lines)
    avg_conf = sum(line.confidence for line in lines) / len(lines)
    return OCRResult(full_text=full_text, average_confidence=avg_conf, lines=lines)


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
