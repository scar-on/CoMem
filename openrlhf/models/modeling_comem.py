"""Public CoMem model entry point.

The implementation remains in ``modeling_clara.py`` for backward
compatibility with existing checkpoints.  This module provides the public
CoMem names used by Hugging Face ``auto_map`` entries.
"""

try:
    from .modeling_clara import CLaRa, CLaRaConfig
except ImportError:  # Hugging Face dynamic module loading
    from modeling_clara import CLaRa, CLaRaConfig


class CoMemConfig(CLaRaConfig):
    """Configuration for CoMem context compression models."""

    model_type = "CoMem"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_map = {
            "AutoConfig": "modeling_comem.CoMemConfig",
            "AutoModel": "modeling_comem.CoMem",
        }


class CoMem(CLaRa):
    """CoMem model with the backward-compatible CLaRa implementation."""

    config_class = CoMemConfig
