import os
from dataclasses import dataclass, field
from typing import Optional

from fairseq.dataclass import FairseqDataclass


@dataclass
class MinIOConfig(FairseqDataclass):
    endpoint: str = field(
        default=os.getenv("MINIO_ENDPOINT", "172.17.0.1:9090"),
        metadata={"help": "MinIO server endpoint"},
    )
    access_key: str = field(
        default=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        metadata={"help": "MinIO access key"},
    )
    secret_key: str = field(
        default=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        metadata={"help": "MinIO secret key"},
    )
    secure: bool = field(
        default=False,
        metadata={"help": "Use secure connection to MinIO server"},
    )
    bucket: str = field(
        default=os.getenv("MINIO_BUCKET", "notturno"),
        metadata={"help": "MinIO bucket name"},
    )
