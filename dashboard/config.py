from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DashboardConfig:
    api_base_url: str = "http://localhost:8000"

    @classmethod
    def from_env(cls) -> DashboardConfig:
        return cls(api_base_url=os.getenv("API_BASE_URL", cls.api_base_url))
