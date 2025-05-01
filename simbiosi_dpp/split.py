from pathlib import Path
import pandas as pd
import numpy as np

def split_by_ids(input_file: Path,
                 output_dir: Path,
                 id_col: str,
                 ratios: list,
                 seed: int = 42):
    """
    Split a TSV file into train, validation, and test sets based on unique IDs.

    Parameters:
    - input_file: Path to the input TSV file.
    - output_dir: Directory where output files will be saved.
    - id_col: Column name to group by.
    - ratios: List of three floats summing to 1.0, e.g. [0.1, 0.8, 0.1] for test, train, val.
    - seed: Random seed for reproducibility.
    """
    # Load data
    df = pd.read_csv(input_file, sep="\t")

    # Ensure ratios sum to 1
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Ratios must sum to 1.0")

    # Get unique IDs and assign random values
    rng = np.random.RandomState(seed)
    unique_ids = np.array(df[id_col].unique())
    rand_vals = rng.rand(len(unique_ids))

    # Determine thresholds
    t_cut = ratios[0]
    tr_cut = ratios[0] + ratios[1]

    # Assign split labels based on random values
    labels = np.where(rand_vals < t_cut, 'test',
                      np.where(rand_vals < tr_cut, 'train', 'val'))

    # Create DataFrame for ID-to-split mapping
    map_df = pd.DataFrame({id_col: unique_ids, 'split': labels})

    # Merge split labels back to main DataFrame
    df = df.merge(map_df, on=id_col)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each split
    base_name = input_file.stem
    for split_name in ['test', 'train', 'val']:
        split_df = df[df['split'] == split_name].drop(columns=['split'])
        out_file = output_dir / f"{base_name}_{split_name}.tsv"
        split_df.to_csv(out_file, sep="\t", index=False)
        print(f"Saved {split_name} set: {out_file} ({len(split_df)} rows)")


if __name__ == "__main__":
    # === 硬编码参数设置 ===
    input_file = Path("data-bin/all_with_ids.tsv")   # 输入 TSV 文件路径，请根据需要修改
    output_dir = Path("data-bin/splits")     # 输出目录，请根据需要修改
    id_col = "id"                             # 用于分组的列名
    ratios = [0.1, 0.8, 0.1]                      # 测试集、训练集、验证集比例
    seed = 42                                     # 随机种子，保证可复现

    # 执行拆分
    split_by_ids(input_file, output_dir, id_col, ratios, seed)