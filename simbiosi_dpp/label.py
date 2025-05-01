from pathlib import Path

import pandas as pd
from tqdm import tqdm

tqdm.pandas()
speakers = {}


def get_person(object_name: str):
    speaker_name = object_name.split("/")[-2]
    return speaker_name


def person_to_id(speaker_name: str):
    global speakers
    if speaker_name not in speakers:
        speakers[speaker_name] = len(speakers)
    return speakers[speaker_name]


def main():
    tsv_path = Path("data-bin/all.tsv")
    output_path = Path("data-bin/all_with_ids.tsv")
    df = pd.read_csv(tsv_path, sep="\t")
    df["person"] = df["object_name"].progress_apply(get_person)
    df["id"] = df["person"].progress_apply(person_to_id)
    df = df.drop(columns=["md5", "size"])
    df = df.sort_values(by=["id"])
    print(df.head())
    df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
