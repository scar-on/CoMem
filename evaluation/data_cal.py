import argparse

from datasets import load_dataset
import pprint

parser = argparse.ArgumentParser(description="Inspect a parquet dataset example.")
parser.add_argument("parquet_file", help="Path to a parquet dataset shard")
args = parser.parse_args()

dataset = load_dataset(
    "parquet",
    data_files=args.parquet_file,
    split="train"
)

example = dataset[0]

pp = pprint.PrettyPrinter(
    width=120,      # 每行最大宽度
    depth=None,     # 不限制嵌套深度
    compact=False   # 不压缩
)

pp.pprint(example)
