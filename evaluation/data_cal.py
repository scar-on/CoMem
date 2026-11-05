from datasets import load_dataset
import pprint

dataset = load_dataset(
    "parquet",
    data_files="/data/project/scar/CLaRa_pretraining/train-00000-of-00021-07780a8f8e707c96.parquet",
    split="train"
)

example = dataset[0]

pp = pprint.PrettyPrinter(
    width=120,      # 每行最大宽度
    depth=None,     # 不限制嵌套深度
    compact=False   # 不压缩
)

pp.pprint(example)
