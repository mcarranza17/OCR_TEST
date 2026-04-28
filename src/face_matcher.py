from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


class FaceMatcherError(RuntimeError):
    pass


@dataclass(frozen=True)
class FaceComparisonResult:
    similarity: float
    decision: str
    document_bbox: list[int]
    selfie_bbox: list[int]
    document_face_ratio: float
    selfie_face_ratio: float
    match_threshold: float
    review_threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FaceMatcher:
    DET_SIZES: tuple[tuple[int, int], ...] = ((640, 640), (1024, 1024), (1600, 1600))
    MIN_LONG_SIDE = 1280

    def __init__(
        self,
        model_name: str = "buffalo_l",
        match_threshold: float = 0.38,
        review_threshold: float = 0.25,
        model_root: str = ".models/insightface",
    ) -> None:
        if review_threshold > match_threshold:
            raise FaceMatcherError("El umbral de revisión no puede ser mayor al umbral de match.")

        self.match_threshold = match_threshold
        self.review_threshold = review_threshold

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise FaceMatcherError("InsightFace no está instalado. Ejecuta pip install -r requirements.txt") from exc

        try:
            self._app = FaceAnalysis(
                name=model_name,
                root=model_root,
                providers=["CPUExecutionProvider"],
            )
            self._current_det_size: tuple[int, int] = self.DET_SIZES[0]
            self._app.prepare(ctx_id=-1, det_size=self._current_det_size)
        except Exception as exc:  # noqa: BLE001 - model loading errors are framework-specific.
            raise FaceMatcherError(f"No se pudo cargar el modelo facial {model_name}: {exc}") from exc

    def compare(self, document_image_bgr: np.ndarray, selfie_image_bgr: np.ndarray) -> FaceComparisonResult:
        doc_face, doc_scale, doc_shape = self._detect_largest(document_image_bgr, "documento")
        selfie_face, selfie_scale, selfie_shape = self._detect_largest(selfie_image_bgr, "selfie")

        doc_embedding = _embedding(doc_face)
        selfie_embedding = _embedding(selfie_face)
        similarity = _cosine_similarity(doc_embedding, selfie_embedding)

        if similarity >= self.match_threshold:
            decision = "match"
        elif similarity >= self.review_threshold:
            decision = "review"
        else:
            decision = "no_match"

        return FaceComparisonResult(
            similarity=similarity,
            decision=decision,
            document_bbox=_bbox_scaled(doc_face, doc_scale),
            selfie_bbox=_bbox_scaled(selfie_face, selfie_scale),
            document_face_ratio=_face_area_ratio(doc_face, doc_scale, doc_shape),
            selfie_face_ratio=_face_area_ratio(selfie_face, selfie_scale, selfie_shape),
            match_threshold=self.match_threshold,
            review_threshold=self.review_threshold,
        )

    def _detect_largest(self, image_bgr: np.ndarray, source_name: str):
        original_shape = image_bgr.shape[:2]
        enhanced, scale = _enhance(image_bgr, self.MIN_LONG_SIDE)

        for candidate, sc in ((enhanced, scale), (image_bgr, 1.0)):
            for size in self.DET_SIZES:
                self._ensure_det_size(size)
                faces = self._app.get(candidate)
                if faces:
                    best = max(faces, key=lambda f: _bbox_area(f.bbox))
                    return best, sc, original_shape

        raise FaceMatcherError(
            f"No se detectó rostro en la imagen de {source_name}. "
            "Mejora iluminación, enfoque y distancia."
        )

    def _ensure_det_size(self, size: tuple[int, int]) -> None:
        if self._current_det_size != size:
            self._app.prepare(ctx_id=-1, det_size=size)
            self._current_det_size = size


def _enhance(image_bgr: np.ndarray, min_long_side: int) -> tuple[np.ndarray, float]:
    import cv2

    h, w = image_bgr.shape[:2]
    long_side = max(h, w)
    scale = 1.0
    img = image_bgr
    if long_side < min_long_side:
        scale = min_long_side / long_side
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return enhanced, scale


def _embedding(face) -> np.ndarray:
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        return np.asarray(face.normed_embedding, dtype=np.float32)
    if hasattr(face, "embedding") and face.embedding is not None:
        return np.asarray(face.embedding, dtype=np.float32)
    raise FaceMatcherError("El modelo no devolvió embedding facial.")


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0 or right_norm == 0:
        raise FaceMatcherError("Embedding facial inválido.")
    return float(np.dot(left, right) / (left_norm * right_norm))


def _bbox_scaled(face, scale: float) -> list[int]:
    if scale == 1.0:
        return [int(value) for value in face.bbox.tolist()]
    return [int(value / scale) for value in face.bbox.tolist()]


def _bbox_area(bbox: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _face_area_ratio(face, scale: float, original_shape: tuple[int, int]) -> float:
    h, w = original_shape
    if h == 0 or w == 0:
        return 0.0
    bbox = face.bbox if scale == 1.0 else (face.bbox / scale)
    area = _bbox_area(bbox)
    return float(area / (h * w))
