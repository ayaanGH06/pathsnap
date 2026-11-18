
"""Environment-based runtime configuration for the PathSnap API."""

import os
from dataclasses import dataclass
from pathlib import Path

VALID_PARTITIONS = ("train", "student_test")


LAN_ORIGIN_REGEX = (
    r"^http://(localhost|127\.0\.0\.1|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+"
    r"|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?$"
)


@dataclass(frozen=True)
class Settings:
    data_root: Path
    partition: str
    allowed_origins: tuple[str, ...]
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def trips_dir(self) -> Path:
        return self.data_root / self.partition


def get_settings() -> Settings:
    data_root = Path(
        os.getenv("PATHSNAP_DATA_ROOT", "data")
    ).expanduser().resolve()

    partition = os.getenv("PATHSNAP_PARTITION", "student_test").strip()
    if partition not in VALID_PARTITIONS:
        raise ValueError(
            f"Invalid partition {partition!r}; expected one of {VALID_PARTITIONS}"
        )

    # Default: no explicit origins; main.py allows localhost and private-LAN
    # origins only (see LAN_ORIGIN_REGEX). Add more with PATHSNAP_CORS_ORIGINS.
    origins = os.getenv("PATHSNAP_CORS_ORIGINS", "")
    return Settings(
        data_root=data_root,
        partition=partition,
        allowed_origins=tuple(
            origin.strip() for origin in origins.split(",") if origin.strip()
        ),
        host=os.getenv("PATHSNAP_API_HOST", "0.0.0.0").strip(),
        port=int(os.getenv("PATHSNAP_API_PORT", "8000")),
    )
