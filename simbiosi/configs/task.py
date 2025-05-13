from dataclasses import dataclass
from typing import Optional

from fairseq.dataclass import FairseqDataclass

from simbiosi.configs.minio import MinIOConfig


@dataclass
class SimBiosiTaskConfig(FairseqDataclass):
    train_tsv: Optional[str] = None
    val_tsv: Optional[str] = None
    minio: MinIOConfig = MinIOConfig()
