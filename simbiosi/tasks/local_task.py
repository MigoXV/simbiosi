import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task

from simbiosi.datasets.local_dataset import LocalMinIODataset, LocalMinIOTrainDataset

logger = logging.getLogger(__name__)


@dataclass
class LocalSimBiosiTaskConfig(FairseqDataclass):
    train_tsv: Optional[str] = None
    val_tsv: Optional[str] = None
    root_dir: Optional[str] = None
    use_augmentation: bool = True


@register_task("simbiosi-task-local", dataclass=LocalSimBiosiTaskConfig)
class LocalSimbiosiTask(FairseqTask):
    def __init__(self, cfg: LocalSimBiosiTaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg = cfg
        self.root_dir = Path(cfg.root_dir)
        self.train_df = pd.read_csv(cfg.train_tsv, sep="\t")
        self.valid_df = pd.read_csv(cfg.val_tsv, sep="\t")
        self.num_classes = max(self.train_df["id"].max(), self.valid_df["id"].max()) + 1

    def load_dataset(self, split, **kwargs):
        if split == "train":
            self.datasets[split] = LocalMinIODataset(
                df=self.train_df,
                root_dir=self.root_dir,
            )
        elif split == "valid":
            self.datasets[split] = LocalMinIOTrainDataset(
                df=self.valid_df,
                root_dir=self.root_dir,
                use_augmentation=self.cfg.use_augmentation,
            )
        else:
            raise ValueError(f"Unknown split: {split}")

    @property
    def target_dictionary(self):
        return None
