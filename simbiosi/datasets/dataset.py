import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from fairseq import metrics
from fairseq.data import FairseqDataset
from violoncello.drivers.httpx_driver import HttpxAudioDriver

from simbiosi.configs.minio import MinIOConfig


class MinIODataset(FairseqDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        driver_config: MinIOConfig,
    ):
        self.df = df
        # init minio
        self.driver = HttpxAudioDriver(
            endpoint=driver_config.endpoint,
            bucket=driver_config.bucket,
            access_key=driver_config.access_key,
            secret_key=driver_config.secret_key,
            secure=driver_config.secure,
            sr=8000,
            session_warmup=10,
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = self.driver.get_object(row["object_name"])
        image = iio.imread(image)
        image = self.transform(image)
        face_id = row["id"]
        return image, face_id

    def collater(self, samples):
        if len(samples) == 0:
            return {}
        images, face_ids = zip(*samples)
        images = torch.stack(images, dim=0)
        face_ids = torch.tensor(face_ids)
        return images, face_ids

    def size(self, index):
        return 1

    def num_tokens(self, index):
        return 1


class MinIOTrainDataset(MinIODataset):

    def set_epoch(self, epoch):
        self.epoch = epoch
        metrics.log_scalar("epoch", epoch)
        metrics.log_scalar("epoch", epoch)
