from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ocr_lang: str = "es"
    face_model_name: str = "buffalo_l"
    face_match_threshold: float = 0.38
    face_review_threshold: float = 0.25
    model_cache_dir: str = ".models"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ocr_lang=os.getenv("OCR_LANG", cls.ocr_lang),
            face_model_name=os.getenv("FACE_MODEL_NAME", cls.face_model_name),
            face_match_threshold=float(os.getenv("FACE_MATCH_THRESHOLD", cls.face_match_threshold)),
            face_review_threshold=float(os.getenv("FACE_REVIEW_THRESHOLD", cls.face_review_threshold)),
            model_cache_dir=os.getenv("MODEL_CACHE_DIR", cls.model_cache_dir),
        )

    @property
    def model_cache_path(self) -> Path:
        return Path(self.model_cache_dir).resolve()

    @property
    def paddlex_cache_home(self) -> str:
        return str(self.model_cache_path / "paddlex")

    @property
    def insightface_root(self) -> str:
        return str(self.model_cache_path / "insightface")

    @property
    def matplotlib_config_dir(self) -> str:
        return str(self.model_cache_path / "matplotlib")

    @property
    def huggingface_home(self) -> str:
        return str(self.model_cache_path / "huggingface")

    @property
    def modelscope_cache_dir(self) -> str:
        return str(self.model_cache_path / "modelscope")

    def configure_runtime(self) -> None:
        self.model_cache_path.mkdir(parents=True, exist_ok=True)
        for path in (
            self.paddlex_cache_home,
            self.insightface_root,
            self.matplotlib_config_dir,
            self.huggingface_home,
            self.modelscope_cache_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", self.paddlex_cache_home)
        os.environ.setdefault("MPLCONFIGDIR", self.matplotlib_config_dir)
        os.environ.setdefault("HF_HOME", self.huggingface_home)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(self.huggingface_home) / "hub"))
        os.environ.setdefault("MODELSCOPE_CACHE", self.modelscope_cache_dir)
        os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
