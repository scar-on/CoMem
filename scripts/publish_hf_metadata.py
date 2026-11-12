#!/usr/bin/env python3
"""Publish CoMem code, model cards, and portable checkpoint metadata."""

import argparse
import io
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_MAP = {
    "AutoConfig": "modeling_comem.CoMemConfig",
    "AutoModel": "modeling_comem.CoMem",
}


def parse_checkpoint(value: str):
    target, separator, source = value.partition("=")
    if not separator or not source:
        raise argparse.ArgumentTypeError("Expected stageN/crRATE=/path/to/checkpoint")
    parts = target.split("/")
    if len(parts) != 2 or parts[0] not in {"stage1", "stage2"} or not parts[1].startswith("cr"):
        raise argparse.ArgumentTypeError("Expected stageN/crRATE=/path/to/checkpoint")
    try:
        rate = int(parts[1][2:])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Compression rate must be an integer") from exc
    if rate <= 0:
        raise argparse.ArgumentTypeError("Compression rate must be positive")
    return target, rate, Path(source)


def portable_config(source: Path, base_model: str, rate: int, stage: str) -> bytes:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    expected_stage = "stage1" if stage == "stage1" else "stage1_2"
    if config.get("compr_rate") != rate or config.get("training_stage") != expected_stage:
        raise ValueError(f"Checkpoint metadata does not match {stage}/cr{rate}: {source}")
    config.update(
        decoder_model_name=base_model,
        compr_base_model_name=base_model,
        model_type="CoMem",
        auto_map=AUTO_MAP,
    )
    return (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--card", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", required=True, type=parse_checkpoint)
    parser.add_argument("--delete-path", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = set()
    configs = []
    manifest = {"base_model": args.base_model, "checkpoints": []}
    for target, rate, source in args.checkpoint:
        if target in paths:
            parser.error(f"Duplicate checkpoint target: {target}")
        paths.add(target)
        for filename in ("adapters.pth", "decoder_first_last_layers.pth", "tokenizer.json"):
            if not (source / filename).is_file():
                parser.error(f"Missing {filename} in {source}")
        stage = target.split("/")[0]
        configs.append((f"{target}/config.json", portable_config(source, args.base_model, rate, stage)))
        manifest["checkpoints"].append({"stage": stage, "compression_rate": rate, "path": target})

    files = [("README.md", args.card), ("LICENSE", REPO_ROOT / "LICENSE"),
             ("ACKNOWLEDGEMENTS", REPO_ROOT / "ACKNOWLEDGEMENTS"),
             ("modeling_comem.py", REPO_ROOT / "openrlhf/models/modeling_comem.py")]
    files.extend((str(path.relative_to(REPO_ROOT)), path) for path in sorted((REPO_ROOT / "openrlhf").rglob("*.py")))
    for target, source in files:
        if not source.is_file():
            parser.error(f"Missing release file: {source}")

    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    planned = [target for target, _ in files + configs] + ["checkpoint_manifest.json"]
    if args.dry_run:
        print(f"Repo: {args.repo_id}")
        print("Add or update:", *planned, sep="\n  ")
        print("Delete:", *args.delete_path, sep="\n  ")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        parser.error("HF_TOKEN must be set")
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    operations = [CommitOperationAdd(path_in_repo=target, path_or_fileobj=source) for target, source in files]
    operations.extend(CommitOperationAdd(path_in_repo=target, path_or_fileobj=io.BytesIO(content)) for target, content in configs)
    operations.append(CommitOperationAdd(path_in_repo="checkpoint_manifest.json", path_or_fileobj=io.BytesIO(manifest_bytes)))
    operations.extend(CommitOperationDelete(path_in_repo=target) for target in args.delete_path)
    commit = HfApi(token=token).create_commit(
        repo_id=args.repo_id,
        repo_type="model",
        operations=operations,
        commit_message="Publish CoMem code and portable checkpoint metadata",
    )
    print(commit.commit_url)


if __name__ == "__main__":
    main()
