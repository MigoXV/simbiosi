import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def remap_ids_by_person(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据 person 列，重新从 0 开始为每个不同的 person 分配连续的 id
    """
    df = df.copy()
    df['id'] = pd.Categorical(df['person']).codes
    return df

def select_validation_set(df: pd.DataFrame, id_count: int = 100, images_per_id: int = 2) -> pd.DataFrame:
    """
    - 找出样本数 > images_per_id 的 id
    - 随机抽取 id_count 个 id
    - 每个 id 随机抽 images_per_id 张图片
    返回的 val_df 会保留原始行索引，便于后续从 df 中删除
    """
    counts = df['id'].value_counts()
    eligible_ids = counts[counts > images_per_id].index.tolist()

    if len(eligible_ids) < id_count:
        raise ValueError(f"满足 > {images_per_id} 张的 id 只有 {len(eligible_ids)} 个，少于所需的 {id_count} 个")

    selected_ids = random.sample(eligible_ids, id_count)

    pieces = []
    for id_ in tqdm(selected_ids, desc="Selecting validation images"):
        sampled = df[df['id'] == id_].sample(n=images_per_id)
        pieces.append(sampled)

    val_df = pd.concat(pieces)  # 保留原始索引
    return val_df

if __name__ == "__main__":
    # ======= 配置部分（硬编码输入/输出路径） =======
    tsv_path    = Path("data-bin/splits/all_with_ids_train.tsv")  # 输入 TSV 文件
    output_dir  = Path("data-bin/splits")            # 输出目录
    # ============================================

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取原始 TSV
    df = pd.read_csv(tsv_path, sep='\t')

    # 2. 根据 person 重新分配连续 id
    df = remap_ids_by_person(df)

    # 3. 生成验证集 DataFrame（100 个 id，每个 id 2 张）
    val_df = select_validation_set(df, id_count=100, images_per_id=2)

    # 4. 剩余部分作为训练集
    train_df = df.drop(val_df.index)

    # 5. 保存到文件，使用 _val 和 _train 后缀
    base_name   = tsv_path.stem
    val_path    = output_dir / f"{base_name}_val.tsv"
    train_path  = output_dir / f"{base_name}_train.tsv"

    # 只保留三列并按需求顺序输出
    cols = ['object_name', 'person', 'id']
    val_df.to_csv(val_path, sep='\t', index=False, columns=cols)
    train_df.to_csv(train_path, sep='\t', index=False, columns=cols)

    print(f"重新分配了 id，共有 {df['id'].nunique()} 个类别")
    print(f"验证集已保存到: {val_path}")
    print(f"训练集已保存到: {train_path}")

