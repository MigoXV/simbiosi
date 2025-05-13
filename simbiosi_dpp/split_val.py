from pathlib import Path
import pandas as pd
import numpy as np

def split_each_id(input_file: Path,
                  output_dir: Path,
                  id_col: str,
                  val_ratio: float,
                  seed: int = 42):
    """
    对每个 person 重新编号 id，然后按 id 分组划分训练集和验证集。

    参数：
    - input_file: 输入 TSV 文件路径。
    - output_dir: 输出目录。
    - id_col: 用于分组的列名（通常是类别 id）。
    - val_ratio: 验证集比例（0~1之间）。
    - seed: 随机种子。
    """
    df = pd.read_csv(input_file, sep="\t")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio 必须在 0 到 1 之间")

    # === 重新编号 id ===
    # 将每个 person 映射到一个新的 id（按字典序或出现顺序均可）
    person2id = {person: i for i, person in enumerate(sorted(df["person"].unique()))}
    df["id"] = df["person"].map(person2id)

    rng = np.random.default_rng(seed)

    train_rows = []
    val_rows = []

    # 针对每个 id 单独划分
    for label, group in df.groupby(id_col):
        indices = group.index.to_numpy()
        rng.shuffle(indices)

        n_val = int(len(indices) * val_ratio)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        val_rows.append(df.loc[val_idx])
        train_rows.append(df.loc[train_idx])

    # 合并所有分组结果
    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)

    # 输出结果
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_file.stem
    train_path = output_dir / f"{base_name}_train.tsv"
    val_path = output_dir / f"{base_name}_val.tsv"

    train_df.to_csv(train_path, sep="\t", index=False)
    val_df.to_csv(val_path, sep="\t", index=False)

    print(f"已重新编号：{len(person2id)} 个类别")
    print(f"训练集：{len(train_df)} 行，保存于 {train_path}")
    print(f"验证集：{len(val_df)} 行，保存于 {val_path}")


if __name__ == "__main__":
    # === 参数设置 ===
    input_file = Path("data-bin/splits2/filter_all_with_ids_train.tsv")
    output_dir = Path("data-bin/splits2")
    id_col = "id"
    val_ratio = 0.2
    seed = 42

    split_each_id(input_file, output_dir, id_col, val_ratio, seed)
