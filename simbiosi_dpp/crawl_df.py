from minio import Minio
from violoncello.crawl import crawl

client = Minio(
    "192.168.0.222:39000",
    access_key="test-dataset",
    secret_key="test-dataset",
    secure=False,
)
df = crawl(
    client=client,
    bucket="simbiosi",
    prefix="lfw/deepfunneled",
)

df.to_csv("data-bin/all.tsv", sep="\t", index=False)
