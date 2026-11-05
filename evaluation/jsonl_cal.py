import json


# 2wiki: 12576
# hotpotqa:7384
# nq:6489
# musique:2417
file_path = "/data/li.zhiheng/scar_workspace/comem/evaluation/evaluation_data/end_to_end_evaluation/2wiki/eval_processed_no_pos.jsonl"
file_path = "/data/li.zhiheng/scar_workspace/comem/evaluation/evaluation_data/end_to_end_evaluation/hotpotqa/eval_processed_no_pos.jsonl"
file_path = "/data/li.zhiheng/scar_workspace/comem/evaluation/evaluation_data/end_to_end_evaluation/nq/eval_processed_no_pos.jsonl"
#file_path = "/data/li.zhiheng/scar_workspace/comem/evaluation/evaluation_data/end_to_end_evaluation/nq/eval_processed_no_pos.jsonl"
count = 0

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # 跳过空行
            count += 1

print(f"总条数: {count}")