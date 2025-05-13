from pathlib import Path
import pandas as pd
import numpy as np

def split_by_ids(input_file: Path,
                 output_dir: Path,
                 id_col: str,
                 test_ratio: float,
                 seed: int = 42):
    """
    根据唯一 ID 对 TSV 文件进行划分为训练集和测试集。

    参数：
    - input_file: 输入的 TSV 文件路径。
    - output_dir: 输出目录。
    - id_col: 用于分组的列名（通常是类别 id）。
    - test_ratio: 测试集比例（0 到 1 之间）。
    - seed: 随机种子，用于结果可复现。
    """
    # 加载数据
    df = pd.read_csv(input_file, sep="\t")

    # 检查 test_ratio 是否合理
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio 必须在 0 到 1 之间")

    # 获取唯一 ID，并为其分配随机值
    rng = np.random.RandomState(seed)
    unique_ids = np.array(df[id_col].unique())
    rand_vals = rng.rand(len(unique_ids))

    # 根据随机值决定每个 ID 属于训练集还是测试集
    labels = np.where(rand_vals < test_ratio, 'test', 'train')

    # 构建 ID 到 split 的映射表
    map_df = pd.DataFrame({id_col: unique_ids, 'split': labels})

    # 把划分结果合并回原始数据中
    df = df.merge(map_df, on=id_col)

    # 创建输出目录（如果不存在）
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存每个子集为独立文件
    base_name = input_file.stem
    for split_name in ['test', 'train']:
        split_df = df[df['split'] == split_name].drop(columns=['split'])
        out_file = output_dir / f"{base_name}_{split_name}.tsv"
        split_df.to_csv(out_file, sep="\t", index=False)
        print(f"已保存 {split_name} 集：{out_file}（共 {len(split_df)} 行）")


if __name__ == "__main__":
    # === 硬编码参数设置 ===
    input_file = Path("data-bin/filter_all_with_ids.tsv")  # 输入文件路径
    output_dir = Path("data-bin/splits2")                  # 输出目录
    id_col = "id"                                           # 用于划分的列（通常是类别 id）
    test_ratio = 0.15                                       # 测试集比例
    seed = 42                                               # 随机种子

    # 执行划分
    split_by_ids(input_file, output_dir, id_col, test_ratio, seed)
