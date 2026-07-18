from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ApiConfig:
    model_bundle_path: str = "artifacts/model_bundle.joblib"
    database_url: str = "postgresql://anomaly:anomaly@localhost:5432/anomaly_detection"
    # Empty for local dev/tests (the app is reachable at the domain root there).
    # Set to "/api" only on EC2's .env, where infra/nginx/anomaly-detection.conf
    # strips the /api/ prefix before forwarding - without this, FastAPI's
    # generated /docs page references /openapi.json as an absolute root path,
    # which Nginx then routes to the dashboard's catch-all instead of the api
    # container, breaking Swagger UI ("does not specify a valid version field").
    root_path: str = ""

    @classmethod
    def from_env(cls) -> ApiConfig:
        return cls(
            model_bundle_path=os.getenv("MODEL_BUNDLE_PATH", cls.model_bundle_path),
            database_url=os.getenv("DATABASE_URL", cls.database_url),
            root_path=os.getenv("API_ROOT_PATH", cls.root_path),
        )
