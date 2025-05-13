from pathlib import Path
import pandas as pd

# 读取CSV
df = pd.read_csv("data-bin/all_with_ids.tsv", sep="\t")

# 统计每个 person 出现的次数
person_counts = df["person"].value_counts()

# 筛选出出现次数在 10 到 20 次之间的 person
valid_persons = person_counts[(person_counts >= 10) & (person_counts <= 20)].index

# 从原数据中筛选这些 person 的行
filtered_df = df[df["person"].isin(valid_persons)].copy()

# 去掉原来的 id 列
filtered_df = filtered_df.drop(columns=["id"])

# 重新编号：为每个 person 分配一个新的 id（按字典序或顺序）
person2id = {name: i for i, name in enumerate(sorted(filtered_df["person"].unique()))}
filtered_df["id"] = filtered_df["person"].map(person2id)

# 打印类别数量
print(f"类别总数: {len(person2id)}")

# 保存
filtered_df.to_csv("data-bin/filter_all_with_ids.tsv", sep="\t", index=False)
