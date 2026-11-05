import torch
from typing import Callable, Optional, Union
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import TransformersKwargs
from transformers.utils.import_utils import _torch_version, is_torch_less_or_equal, is_torchdynamo_compiling

from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


from typing import Optional, Union

import torch
from packaging import version

from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention


# ---------------------------------------------------------------------------
# Module-level storage for query relevance bias (used by flex_attention score_mod)
# Set before encoder forward, cleared after.  Decoder forward sees None.
# ---------------------------------------------------------------------------
_query_relevance_bias: Optional[torch.Tensor] = None   # [B, S]
_query_relevance_bias_scale: float = 1.0
_original_flex_attention_forward = ALL_ATTENTION_FUNCTIONS.get("flex_attention")


def set_query_relevance_bias(bias: torch.Tensor, scale: float = 1.0):
    """Activate query relevance bias for subsequent flex_attention calls."""
    global _query_relevance_bias, _query_relevance_bias_scale
    _query_relevance_bias = bias
    _query_relevance_bias_scale = scale


def clear_query_relevance_bias():
    """Deactivate query relevance bias."""
    global _query_relevance_bias, _query_relevance_bias_scale
    _query_relevance_bias = None
    _query_relevance_bias_scale = 1.0


def register_flex_attention_with_relevance_bias():
    """
    Register the custom flex attention forward into Transformers.

    We patch the global attention-function registry so every model configured with
    `attn_implementation="flex_attention"` goes through our wrapper. The encoder
    path enables relevance bias via `set_query_relevance_bias(...)`, while the
    decoder path remains unchanged because the bias is cleared / left as None.
    """
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex_attention_forward


def create_encoder_block_mask(attention_mask, n_mem_tokens, device):
    """
    Create a BlockMask with causal query+context and bidirectional mem tokens.

    Sequence layout: [padding | query+context | mem_tokens (last n_mem_tokens)]

    - query+context tokens attend causally to each other.
    - mem tokens attend bidirectionally to each other.
    - mem tokens can attend to all query+context tokens.
    - query+context tokens cannot attend to mem tokens.

    Args:
        attention_mask: 2D padding mask [B, S], 1=real token, 0=padding
        n_mem_tokens: number of memory tokens appended at the end of each sequence
        device: torch device

    Returns:
        BlockMask for flex_attention
    """
    B, S = attention_mask.shape
    mem_start = S - n_mem_tokens
    valid_start = attention_mask.int().argmax(dim=1).to(device)
    valid_len = attention_mask.sum(dim=1).to(device)

    def mask_mod(b, h, q_idx, kv_idx):
        q_is_mem = q_idx >= mem_start
        kv_is_mem = kv_idx >= mem_start
        kv_valid = (kv_idx >= valid_start[b]) & (kv_idx < valid_start[b] + valid_len[b])

        # query+context region: causal, and cannot attend to mem tokens
        qc_mask = ~q_is_mem & ~kv_is_mem & (q_idx >= kv_idx)
        # mem region: bidirectional among mem tokens, and can attend to all query+context tokens
        mem_mask = q_is_mem & (kv_is_mem | ~kv_is_mem)

        return (qc_mask | mem_mask) & kv_valid

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device)

# def create_encoder_block_mask(attention_mask, n_mem_tokens, device):
#     B, S = attention_mask.shape
#     valid_start = attention_mask.int().argmax(dim=1).to(device)
#     valid_len = attention_mask.sum(dim=1).to(device)

#     def mask_mod(b, h, q_idx, kv_idx):
#         kv_valid = (kv_idx >= valid_start[b]) & (kv_idx < valid_start[b] + valid_len[b])
#         causal = q_idx >= kv_idx
#         return causal & kv_valid

#     block_mask = create_block_mask(
#         mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device, _compile=True
#     )
#     return block_mask

    
def create_decoder_block_mask(attention_mask, device):
    """
    Create a BlockMask for the decoder with causal masking and padding support.

    Each token can only attend to itself and preceding non-padding tokens.
    Works regardless of padding side (left or right).

    Uses scalar comparisons (not 2D tensor indexing) because create_block_mask
    rounds indices up to the block size, which would cause out-of-bounds access
    on the original attention_mask tensor.

    Args:
        attention_mask: 2D padding mask [B, S], 1=real token, 0=padding
        device: torch device

    Returns:
        BlockMask for flex_attention
    """
    B, S = attention_mask.shape
    valid_start = attention_mask.int().argmax(dim=1).to(device)
    valid_len = attention_mask.sum(dim=1).to(device)

    def mask_mod(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        kv_valid = (kv_idx >= valid_start[b]) & (kv_idx < valid_start[b] + valid_len[b])
        return causal & kv_valid

    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed




# Flex Attention Replacement
class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self, training):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled or training != self.training:
            self.training = training
            if is_torch_less_or_equal("2.5.1"):
                self._compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
            # In PyTorch 2.6.0, there's a known issue with flex attention compilation which may
            # cause errors. The suggested fix is to compile with "max-autotune-no-cudagraphs"
            # see https://github.com/pytorch/pytorch/issues/146260 for training
            elif version.parse(_torch_version).base_version == "2.6.0" and training:
                self._compiled_flex_attention = torch.compile(
                    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
                )
            # Fallback, usually the most recent torch 2.7.x+ versions
            else:
                self._compiled_flex_attention = torch.compile(flex_attention)

            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    training=False,
    **kwargs,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    # Do not use compiled version if already compiling forward (it raises issues)
    flex_attention_compiled = WrappedFlexAttention(training)() if not is_torchdynamo_compiling() else flex_attention
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    head_mask: Optional[torch.Tensor] = None,
    s_aux: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

    if kwargs.get("dropout", 0.0) > 0:
        raise ValueError(
            "`flex_attention` does not support `dropout`. Please use it with inference"
            " only (`model.eval()`) or turn off the attention dropout in the respective config."
        )

    block_mask = None
    score_mask = None
    if isinstance(attention_mask, BlockMask):
        block_mask = attention_mask
    else:
        score_mask = attention_mask

    if score_mask is not None:
        score_mask = score_mask[:, :, :, : key.shape[-2]]

    # Capture current relevance bias (set by encoder, None during decoder).
    # Following the ALiBi pattern: pre-computed float32 bias tensor [B, S],
    # indexed with scalar-tensor indices inside score_mod.
    relevance_bias = _query_relevance_bias          # None or [B, S] float32
    relevance_scale = _query_relevance_bias_scale   # float scalar
    # print("relevance_bias:,",relevance_bias)
    if relevance_bias is not None:
        def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
            score = score + relevance_bias[batch_idx, kv_idx] * relevance_scale
            if softcap is not None:
                score = softcap * torch.tanh(score / softcap)
            if score_mask is not None:
                score = score + score_mask[batch_idx][0][q_idx][kv_idx]
            if head_mask is not None:
                score = score + head_mask[batch_idx][head_idx][0][0]
            return score
    else:
        def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
            if softcap is not None:
                score = softcap * torch.tanh(score / softcap)
            if score_mask is not None:
                score = score + score_mask[batch_idx][0][q_idx][kv_idx]
            if head_mask is not None:
                score = score + head_mask[batch_idx][head_idx][0][0]
            return score

    enable_gqa = True
    num_local_query_heads = query.shape[1]

    # When running TP this helps:
    if (num_local_query_heads & (num_local_query_heads - 1)) != 0:
        key = repeat_kv(key, query.shape[1] // key.shape[1])
        value = repeat_kv(value, query.shape[1] // value.shape[1])
        enable_gqa = False

    kernel_options = kwargs.get("kernel_options")
    # On CPU we must skip returning LSE due to a runtime issue; elsewhere, follow PyTorch API and return it
    return_lse = query.device.type != "cpu"

    if not return_lse and s_aux is not None:
        raise ValueError(
            "Attention sinks cannot be run on CPU with flex attention. Please switch to a different device, e.g. CUDA"
        )

    flex_attention_output = compile_friendly_flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        enable_gqa=enable_gqa,
        scale=scaling,
        kernel_options=kernel_options,
        # Last time checked on PyTorch == 2.5.1: Flex Attention always computes the lse regardless.
        # For simplification, we thus always return it as no additional computations are introduced.
        return_lse=return_lse,
        training=module.training,
    )
    # lse is returned in float32
    if return_lse:
        attention_output, lse = flex_attention_output  # type: ignore[misc]
        lse = lse.to(value.dtype)

        if s_aux is not None:
            # Apply attention sinks by renormalizing using LSE
            batch_size, num_heads, seq_len_q, _ = attention_output.shape  # batch, num_heads, seq_len, head_dim
            sinks = s_aux.view(1, -1, 1, 1).expand(batch_size, num_heads, seq_len_q, 1)

            # We need to compute the normalization that includes the sinks
            # since log(sum(exp(scores))) = lse, exp(log(sum(exp(scores)))) = exp(lse)
            # NB: log(sum(exp(scores)) + exp(sink)) = log(exp(lse) + exp(sink))
            lse_expanded = lse.unsqueeze(-1)  # [batch, num_heads, seq_len, 1]
            combined_lse = torch.logsumexp(torch.cat([lse_expanded, sinks], dim=-1), dim=-1, keepdim=True)

            # Use new_norm / old_norm = exp(combined_lse - lse) to compute renorm and apply
            renorm_factor = torch.exp(lse_expanded - combined_lse)
            attention_output = attention_output * renorm_factor
    else:
        attention_output = flex_attention_output  # type: ignore[assignment]
        lse = None

    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, lse


from types import MethodType
from functools import partial

# CUDA_VISIBLE_DEVICES

def modify_method_of_instance(instance, target_class_name, target_method_name, new_method, visited_instances=None):
    """
        This function modifies the method of an instance of a model class. 
        It's part from chat-GPT.
        It will replace the method  with the new method.
        Currently, we only use this function to modify the attention method of a model. Do not test it further. 

        instance: 
            instance of a model to modify.
        target_class_name: 
            name of the attention class to modify. E.g. 'LlamaAttention', 'GPTNeoXAttention', etc.
        new_method: new method to replace the original method. E.g. 'self_extend_forward'. 
            It should include a parameter 'self' to be binded to the instance.
    """
    target_found = False
    if visited_instances is None:
        visited_instances = set()
    # Unique identifier for the instance (using id() since object's id is unique)
    instance_id = id(instance)
    if instance_id in visited_instances:
        target_found = False
        return target_found
    # Add the instance to the already_visited set
    visited_instances.add(instance_id)

    # Check if this instance is of the target class
    if instance.__class__.__name__ == target_class_name:
        bond_method = MethodType(new_method, instance) 
        setattr(instance, target_method_name, bond_method)
        target_found = True
        return target_found
    elif hasattr(instance, '__dict__'):
        for attr_name, attr_value in instance.__dict__.items():
            if isinstance(attr_value, object) and not isinstance(attr_value, (list, tuple, dict, set)):
                _found = modify_method_of_instance(attr_value, target_class_name, target_method_name, new_method, visited_instances)
                if _found:
                    target_found = True
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a dictionary, iterate over its values and recurse
            # E.g, for a ModuleList, its moudels are stored in a dictionary: ._modules
            elif isinstance(attr_value, dict):
                for key, value in attr_value.items():
                    if isinstance(value, object):
                        _found = modify_method_of_instance(value, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True
            # If attribute value is a set, iterate and recurse
            elif isinstance(attr_value, set):
                for item in attr_value:
                    if isinstance(item, object):
                        _found = modify_method_of_instance(item, target_class_name, target_method_name, new_method, visited_instances)
                        if _found:
                            target_found = True

    return target_found

