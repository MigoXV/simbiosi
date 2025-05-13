import pandas as pd
from violoncello.upload import get_df, upload
from pathlib import Path
from yarl import URL
if __name__ == "__main__":
    input_dir = Path("data-bin/lfw-deepfunneled")
    base_url = URL("http://192.168.0.222:39000")
    df = get_df(input_dir, input_dir, disable_tqdm=False)
    df.to_csv("data-bin/lfw-deepfunneled.tsv", sep="\t", index=False)

    upload(
        df,
        base_url,
        bucket="simbiosi",
        minio_prefix="lfw/deepfunneled",
    )