import logging

import pandas as pd
from fairseq.tasks import FairseqTask, register_task

from simbiosi.configs.task import SimBiosiTaskConfig
from simbiosi.datasets.dataset import MinIODataset, MinIOTrainDataset

logger = logging.getLogger(__name__)


@register_task("simbiosi-task", dataclass=SimBiosiTaskConfig)
class SimbiosiTask(FairseqTask):
    def __init__(self, cfg: SimBiosiTaskConfig, **kwargs):
        super().__init__(cfg, **kwargs)
        self.cfg = cfg
        self.train_df = pd.read_csv(cfg.train_tsv, sep="\t")
        self.valid_df = pd.read_csv(cfg.val_tsv, sep="\t")
        self.num_classes = max(self.train_df["id"].max(), self.valid_df["id"].max()) + 1

    def load_dataset(self, split, **kwargs):
        if split == "train":
            self.datasets[split] = MinIOTrainDataset(
                df=self.train_df,
                driver_config=self.cfg.minio,
            )
        elif split == "valid":
            self.datasets[split] = MinIODataset(
                df=self.valid_df,
                driver_config=self.cfg.minio,
            )
        else:
            raise ValueError(f"Unknown split: {split}")

    @property
    def target_dictionary(self):
        return None
