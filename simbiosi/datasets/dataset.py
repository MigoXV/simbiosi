import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from fairseq import metrics
from fairseq.data import FairseqDataset
from PIL import Image
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

    def __init__(self, df, driver_config,use_augmentation:bool=True):
        super().__init__(df, driver_config)
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),  # 接受 numpy array -> PIL.Image
                    transforms.Resize((128, 128)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0)),
                    transforms.Resize((112, 112)),  # 最终人脸尺寸
                    transforms.ToTensor(),  # PIL -> [0,1] FloatTensor, shape=(C,H,W)
                    transforms.Lambda(
                        lambda x: x + 0.01 * torch.randn_like(x)
                    ),  # 加轻微高斯噪声
                    transforms.RandomErasing(
                        p=0.3,
                        scale=(0.02, 0.1),
                        ratio=(0.3, 3.3),
                        value=0,  # 也可以设置成 'random'
                    ),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def set_epoch(self, epoch):
        self.epoch = epoch
        metrics.log_scalar("epoch", epoch)
        metrics.log_scalar("epoch", epoch)
