from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm import tqdm


def get_top_image_shapes(directory: str, top_n: int = 5):
    """
    统计目录中所有图片的尺寸，并打印出现次数最多的n种。
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    shape_counter = Counter()

    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"目录不存在：{directory}")
        return

    image_files = [
        f for f in directory_path.rglob("*") if f.suffix.lower() in image_extensions
    ]

    print(f"\n共找到 {len(image_files)} 张图片，开始统计尺寸...")

    for file_path in tqdm(image_files, desc="统计图片尺寸"):
        try:
            with Image.open(file_path) as img:
                shape = img.size  # (width, height)
                shape_counter[shape] += 1
        except Exception as e:
            tqdm.write(f"无法读取文件 {file_path}: {e}")

    print(f"\n出现次数最多的前 {top_n} 种图片尺寸：")
    for shape, count in shape_counter.most_common(top_n):
        print(f"尺寸 {shape[0]}x{shape[1]}：{count} 张")


def count_image_formats(directory: str):
    """
    统计目录中各种图片文件格式的数量。
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    format_counter = Counter()

    directory_path = Path(directory)
    if not directory_path.exists():
        print(f"目录不存在：{directory}")
        return

    all_files = list(directory_path.rglob("*"))

    for file_path in tqdm(all_files, desc="统计文件格式"):
        ext = file_path.suffix.lower()
        if ext in image_extensions:
            format_counter[ext] += 1

    print("\n图片格式统计结果：")
    for ext, count in format_counter.most_common():
        print(f"{ext.upper()}: {count} 张")


if __name__ == "__main__":
    # 修改为你自己的图片目录
    img_dir = "data-bin/lfw-deepfunneled"

    get_top_image_shapes(img_dir, top_n=5)
    count_image_formats(img_dir)
