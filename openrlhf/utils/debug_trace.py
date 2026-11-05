import hashlib
import json
import os
import platform
import threading
from datetime import datetime
from typing import Any, Dict, Optional

import torch


def _preview_text(text: str, max_chars: int) -> Dict[str, Any]:
    text = text if isinstance(text, str) else str(text)
    preview = text[:max_chars]
    if len(text) > max_chars:
        preview += "...(truncated)"
    return {
        "length": len(text),
        "preview": preview,
    }


class DebugTracer:
    """Small JSONL/PT tracer for cross-machine reproducibility checks."""

    def __init__(
        self,
        trace_dir: str,
        rank: int = 0,
        max_steps: int = 3,
        max_samples: int = 2,
        max_text_chars: int = 240,
    ) -> None:
        self.trace_dir = trace_dir
        self.rank = rank
        self.max_steps = max_steps
        self.max_samples = max_samples
        self.max_text_chars = max_text_chars
        self.trace_file = os.path.join(trace_dir, f"trace_rank{rank}.jsonl")
        self.session_file = os.path.join(trace_dir, f"session_rank{rank}.json")
        self._lock = threading.Lock()
        os.makedirs(trace_dir, exist_ok=True)

    def should_trace(self, step: Optional[int]) -> bool:
        if step is None:
            return True
        return step <= self.max_steps

    def sanitize(self, value: Any, depth: int = 0) -> Any:
        if depth > 4:
            return repr(value)

        if value is None or isinstance(value, (bool, int, float)):
            return value

        if isinstance(value, str):
            return _preview_text(value, self.max_text_chars)

        if isinstance(value, torch.Tensor):
            return self.summarize_tensor(value)

        if isinstance(value, dict):
            return {str(k): self.sanitize(v, depth + 1) for k, v in list(value.items())[:32]}

        if isinstance(value, (list, tuple)):
            items = list(value)
            preview = [self.sanitize(item, depth + 1) for item in items[: self.max_samples]]
            result: Dict[str, Any] = {"length": len(items), "items": preview}
            if len(items) > self.max_samples:
                result["truncated"] = True
            return result

        return repr(value)

    def summarize_tensor(self, tensor: Optional[torch.Tensor], max_values: int = 16) -> Any:
        if tensor is None:
            return None

        data = tensor.detach().cpu().contiguous()
        hash_data = data
        if hash_data.is_floating_point():
            hash_data = hash_data.float()

        checksum = hashlib.sha256(hash_data.numpy().tobytes()).hexdigest()[:16]
        flat = data.reshape(-1)
        preview_tensor = flat[:max_values]
        preview_list = preview_tensor.tolist()

        summary: Dict[str, Any] = {
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "numel": int(data.numel()),
            "checksum": checksum,
            "preview": preview_list,
        }

        if data.numel() > 0 and data.dtype != torch.bool:
            stats = data.float()
            summary.update(
                {
                    "mean": float(stats.mean().item()),
                    "std": float(stats.std(unbiased=False).item()),
                    "min": float(stats.min().item()),
                    "max": float(stats.max().item()),
                    "sum": float(stats.sum().item()),
                    "l2": float(torch.linalg.vector_norm(stats).item()),
                }
            )

        return summary

    def log(self, stage: str, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.should_trace(step):
            return

        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "rank": self.rank,
            "step": step,
            "stage": stage,
            "payload": self.sanitize(payload),
        }
        with self._lock:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def save_artifact(self, name: str, payload: Dict[str, Any], step: Optional[int] = None) -> Optional[str]:
        if not self.should_trace(step):
            return None

        safe_name = name.replace("/", "_").replace(" ", "_")
        prefix = f"step_{step:06d}" if step is not None else "setup"
        artifact_path = os.path.join(self.trace_dir, f"{prefix}_{safe_name}_rank{self.rank}.pt")
        torch.save(payload, artifact_path)
        self.log("artifact", {"name": name, "path": artifact_path}, step=step)
        return artifact_path

    def write_session(self, payload: Dict[str, Any]) -> None:
        with self._lock:
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(self.sanitize(payload), f, indent=2, ensure_ascii=True)


def make_debug_tracer(args, strategy=None) -> Optional[DebugTracer]:
    if not getattr(args, "debug_trace", False):
        return None

    rank = 0
    if strategy is not None and hasattr(strategy, "get_rank"):
        rank = int(strategy.get_rank())

    trace_dir = getattr(args, "debug_trace_dir", None) or os.path.join(args.save_path, "debug_trace")
    tracer = DebugTracer(
        trace_dir=trace_dir,
        rank=rank,
        max_steps=getattr(args, "debug_trace_steps", 3),
        max_samples=getattr(args, "debug_trace_samples", 2),
        max_text_chars=getattr(args, "debug_trace_text_chars", 240),
    )
    tracer.write_session(
        {
            "argv": vars(args),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ],
        }
    )
    tracer.log(
        "session.start",
        {
            "trace_dir": trace_dir,
            "max_steps": tracer.max_steps,
            "max_samples": tracer.max_samples,
        },
        step=None,
    )
    return tracer
