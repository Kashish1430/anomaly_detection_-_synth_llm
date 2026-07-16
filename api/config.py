from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ApiConfig:
    model_bundle_path: str = "artifacts/model_bundle.joblib"

    @classmethod
    def from_env(cls) -> ApiConfig:
        return cls(model_bundle_path=os.getenv("MODEL_BUNDLE_PATH", cls.model_bundle_path))
