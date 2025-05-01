from pathlib import Path
import pandas as pd
import numpy as np

def split_by_ids(input_file: Path,
                 output_dir: Path,
                 id_col: str,
                 test_ratio: float,
                 seed: int = 42):
    """
    Split a TSV file into train and test sets based on unique IDs.

    Parameters:
    - input_file: Path to the input TSV file.
    - output_dir: Directory where output files will be saved.
    - id_col: Column name to group by.
    - test_ratio: Float between 0 and 1, proportion of data to use as test set.
    - seed: Random seed for reproducibility.
    """
    # Load data
    df = pd.read_csv(input_file, sep="\t")

    # Validate test_ratio
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1")

    # Get unique IDs and assign random values
    rng = np.random.RandomState(seed)
    unique_ids = np.array(df[id_col].unique())
    rand_vals = rng.rand(len(unique_ids))

    # Assign split labels based on random values
    labels = np.where(rand_vals < test_ratio, 'test', 'train')

    # Create DataFrame for ID-to-split mapping
    map_df = pd.DataFrame({id_col: unique_ids, 'split': labels})

    # Merge split labels back to main DataFrame
    df = df.merge(map_df, on=id_col)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each split
    base_name = input_file.stem
    for split_name in ['test', 'train']:
        split_df = df[df['split'] == split_name].drop(columns=['split'])
        out_file = output_dir / f"{base_name}_{split_name}.tsv"
        split_df.to_csv(out_file, sep="\t", index=False)
        print(f"Saved {split_name} set: {out_file} ({len(split_df)} rows)")


if __name__ == "__main__":
    # === 硬编码参数设置 ===
    input_file = Path("data-bin/all_with_ids.tsv")   # 输入 TSV 文件路径，请根据需要修改
    output_dir = Path("data-bin/splits")             # 输出目录，请根据需要修改
    id_col = "id"                                    # 用于分组的列名
    test_ratio = 0.15                                 # 测试集比例，训练集自动为 1 - test_ratio
    seed = 42                                        # 随机种子，保证可复现

    # 执行拆分
    split_by_ids(input_file, output_dir, id_col, test_ratio, seed)
