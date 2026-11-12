#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import warnings
import os
import torch
import gc
import time
import json
import copy
import random
import requests
import re

from torch import nn
from torch.nn import functional as F
from torch.nn.functional import gelu
from jinja2.exceptions import TemplateError
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PretrainedConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from huggingface_hub import hf_hub_download
from typing import List, Dict, Any, Optional, Tuple
from openrlhf.models.replace_forward import (
    create_encoder_block_mask,
    create_decoder_block_mask,
    set_query_relevance_bias,
    clear_query_relevance_bias,
    register_flex_attention_with_relevance_bias,
)

# Environment setup
torch.set_printoptions(threshold=float("inf"))
os.environ["NCCL_TIMEOUT"] = "5400"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Constants
IGNORE_INDEX = -100
PARAPHRASE_INSTRUCTIONS = [
    'Background: {docs} means the same as',
    "Background: {docs} Can you put the above sentences in your own terms?",
    "Background: {docs} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {docs}\n(2)",
    "Background: {docs} is a paraphrase of what?",
    "Background: {docs} Could you give me a different version of the background sentences above?",
    "In other words, background: {docs} is just another way of saying:",
    "You're getting across the same point whether you say background: {docs} or",
    "Background: {docs} After unpacking the ideas in the background information above, we got:",
    "Background: {docs} Please offer a restatement of the background sentences I've just read.",
    "Background: {docs}, which also means:",
    "Strip away the mystery, and you'll find background: {docs} is simply another rendition of:",
    "The essence of background: {docs} is captured again in the following statement:",
]


class StopOnCriteria(StoppingCriteria):
    """Custom stopping criteria for generation."""

    def __init__(self, tokenizer, stop_strings: List[str] = None, stop_token_ids: List[int] = None):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings or []
        self.stop_token_ids = stop_token_ids or []
        self.reason = None

    def __call__(self, input_ids, scores, **kwargs):
        # Check if last token is in stop_token_ids
        last_token = input_ids[0, -1].item()
        if last_token in self.stop_token_ids:
            self.reason = f"stop_token_{last_token}"
            return True

        # Check if any stop_strings appear in generated text
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        for stop_str in self.stop_strings:
            if stop_str in text:
                self.reason = f"stop_string_{stop_str}"
                return True

        return False


class LlamaRMSNorm(nn.Module):
    """Llama-style RMS normalization layer."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class ContrastProjectionHead(nn.Module):
    """
    Two-layer MLP projection head for contrastive learning.

    Applied token-wise to each mem-token embedding BEFORE mean pooling,
    keeping input and output dimensions equal to hidden_size.
    Architecture: Linear(D, hidden_dim) → GELU → Linear(hidden_dim, D)

    The projection head is used only during contrastive training and is NOT
    used for downstream decoding.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim]  — works for both 2-D and 3-D inputs
        return self.net(x)


class Converter(nn.Module):
    """Converter module for dimension transformation."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.rms_norm = LlamaRMSNorm(input_dim)
        self.dense_in = nn.Linear(input_dim, output_dim)
        self.dense_out = nn.Linear(output_dim, output_dim)

        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        """Print parameter statistics."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Converter trainable parameters: {trainable_params}, Total parameters: {total_params}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self.rms_norm(embeddings)
        x = self.dense_in(embeddings)
        x = self.dense_out(gelu(x))
        return x.to(torch.float32)


class CoMemConfig(PretrainedConfig):
    """Configuration class for CoMem model."""

    model_type = "CoMem"

    def __init__(self,
                 decoder_model_name: str = "meta-llama/Llama-2-7b-chat-hf",
                 doc_max_length: int = 128,
                 quantization: str = 'no',
                 sep: bool = False,
                 compr_model_name: str = "google-bert/bert-base-uncased",
                 compr_rate: int = 64,
                 compr_n_layers: int = None,
                 compr_every_n_layer: int = None,
                 compr_base_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 compr_rms_norm: bool = False,
                 compr_mlp_hidden_dim: int = 8096,
                 compr_use_mlp: bool = True,
                 compr_linear_type: str = "concat",
                 lora: bool = False,
                 lora_compressor: bool = False,
                 training_form: str = "both",
                 training_stage: str = "stage1",
                 generation_top_k: int = 1,
                 lora_r: int = 16,
                 lora_r_compressor: int = None,
                 load_adapters: bool = True,
                 kbtc_training: bool = False,
                 optimize_mem_tokens: bool = False,
                 different_mem_tokens: bool = False,
                 attn_implementation: str = None,
                 _attn_implementation_autoset: bool = True,
                 ae_mode: str = "token",
                 max_new_tokens: int = 128,
                 load_pretrained_checkpoint: bool = False,
                 device_map=None,
                 use_mean_pool: bool = False, # add
                 mean_pool_dropout: float = 0.1, # add 0.1 to 0.3
                 simcse_temperature: float = 0.05, # add
                 use_proj_head: bool = False,  # contrastive projection head
                 proj_hidden_ratio: int = 1,   # projection head intermediate dim = hidden_size * ratio
                 use_query_relevance_bias: bool = False,
                 relevance_bias_scale: float = 1.0,
                 simcse_in_batch: bool = False,
                 ortho_loss_weight: float = 0.0,
                 train_encoder_adapter: bool = True,
                 train_decoder_adapter: bool = True,
                 pure_inference: bool = False,
                 auto_map: dict = {
                     "AutoConfig": "modeling_comem.CoMemConfig",
                     "AutoModel": "modeling_comem.CoMem"
                 },
                 **kwargs):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name
        self.doc_max_length = doc_max_length
        self.quantization = quantization
        self.sep = sep

        self.compr_model_name = compr_model_name
        self.compr_rate = compr_rate
        self.compr_use_mlp = compr_use_mlp
        self.compr_mlp_hidden_dim = compr_mlp_hidden_dim
        self.compr_n_layers = compr_n_layers
        self.compr_every_n_layer = compr_every_n_layer
        self.compr_base_model_name = compr_base_model_name
        self.compr_rms_norm = compr_rms_norm
        self.compr_linear_type = compr_linear_type

        self.lora = lora
        self.lora_compressor = lora_compressor
        self.training_form = training_form
        self.lora_r = lora_r
        self.lora_r_compressor = lora_r_compressor or lora_r
        self.load_adapters = load_adapters
        self.optimize_mem_tokens = optimize_mem_tokens
        self.different_mem_tokens = different_mem_tokens
        self.kbtc_training = kbtc_training
        self.training_stage = training_stage
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self._attn_implementation_autoset = _attn_implementation_autoset
        self.ae_mode = ae_mode
        self.max_new_tokens = max_new_tokens
        self.auto_map = auto_map
        self.load_pretrained_checkpoint = load_pretrained_checkpoint

        self.generation_top_k = generation_top_k

        # Mean pooling and SimCSE settings
        self.use_mean_pool = use_mean_pool
        self.mean_pool_dropout = mean_pool_dropout
        self.simcse_temperature = simcse_temperature
        self.use_proj_head = use_proj_head
        self.proj_hidden_ratio = proj_hidden_ratio
        self.use_query_relevance_bias = use_query_relevance_bias
        self.relevance_bias_scale = relevance_bias_scale
        self.simcse_in_batch = simcse_in_batch
        self.ortho_loss_weight = ortho_loss_weight
        self.train_encoder_adapter = train_encoder_adapter
        self.train_decoder_adapter = train_decoder_adapter
        self.pure_inference = pure_inference

        if training_form == 'compressor':
            assert compr_model_name is not None and not self.lora


# Utility functions
def remote_generate(docs: List[str], questions: List[str], api_url: str) -> List[str]:
    """Generate responses using remote API."""
    response = requests.post(
        f"{api_url}/generate",
        json={"docs": docs, "questions": questions}
    )
    return response.json()["texts"]


def add_memory_tokens_to_inputs(input_ids: torch.Tensor,
                               attention_mask: torch.Tensor,
                               n_mem_tokens: int,
                               tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add memory tokens to input sequences."""
    assert len(tokenizer.mem_tokens) == n_mem_tokens

    mem_tokens = torch.stack([tokenizer.mem_token_ids_pt] * input_ids.size(0), 0)
    assert len(mem_tokens) == input_ids.size(0)
    assert len(mem_tokens[0]) == n_mem_tokens

    input_ids = torch.cat([input_ids, mem_tokens], dim=1)
    attention_mask = torch.cat([attention_mask, torch.ones(input_ids.size(0), n_mem_tokens)], dim=1)

    return input_ids, attention_mask


class CoMem(PreTrainedModel):
    """CoMem: Unified Retrieval-Augmented Generation Model."""

    config_class = CoMemConfig

    def __init__(self, cfg: CoMemConfig):
        super().__init__(cfg)
        register_flex_attention_with_relevance_bias()
        self.decoder_model_name = cfg.decoder_model_name
        self.decoder = self._create_decoder(cfg)
        self.doc_max_length = cfg.doc_max_length

        print(f'Base decoder parameters: {self.decoder.num_parameters()}')

        # Model configuration
        self.compr_model_name = cfg.compr_model_name
        self.training_form = cfg.training_form
        self.lora = cfg.lora
        self.adapter_keys = []
        self.train_encoder_adapter = getattr(cfg, 'train_encoder_adapter', True)
        self.train_decoder_adapter = getattr(cfg, 'train_decoder_adapter', True)
        self._adapter_param_cache = {}
        self.compr = None

        # Initialize LoRA adapters if needed
        if cfg.lora and not getattr(cfg, 'pure_inference', False) and not getattr(cfg, 'delay_adapter_init', False):
            self._setup_lora_adapters(cfg)

        print(f'Model adapter keys: {self.adapter_keys}')

        # Initialize tokenizer and resize embeddings
        self.decoder_tokenizer = self._create_decoder_tokenizer(cfg)
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self._configure_generation_config()

        # Model parameters
        self.generation_top_k = cfg.generation_top_k
        self.training_stage = cfg.training_stage
        self.sep = cfg.sep
        self.compr_rate = cfg.compr_rate
        self.local_rank = os.getenv('LOCAL_RANK', '0')

        self.n_mem_tokens = self.doc_max_length // self.compr_rate
        self.hidden_size = self.decoder.config.hidden_size

        # Setup adapters and memory token optimization
        if self.lora:
            self._setup_adapter_training()
        else:
            print(f'Total trainable parameters: {self.num_parameters(only_trainable=True)}')

        self._prepare_mem_tokens_optimization()
        self._enforce_adapter_trainability(verbose=True)

        # Retrieval configuration
        self.url_retrieval = "http://127.0.0.1:5004/queries"

        # Mean pooling and contrastive learning configuration
        self.use_mean_pool = cfg.use_mean_pool
        self.simcse_temperature = cfg.simcse_temperature
        self.simcse_in_batch = getattr(cfg, 'simcse_in_batch', False)
        self.ortho_loss_weight = getattr(cfg, 'ortho_loss_weight', 0.0)

        # Projection head configuration
        self.use_proj_head = getattr(cfg, 'use_proj_head', False)
        self.proj_hidden_ratio = getattr(cfg, 'proj_hidden_ratio', 1)

        # Query relevance bias for FlexAttention score_mod
        self.use_query_relevance_bias = getattr(cfg, 'use_query_relevance_bias', False)
        self.relevance_bias_scale = getattr(cfg, 'relevance_bias_scale', 1.0)
        if self.use_query_relevance_bias:
            print(f'Query relevance bias enabled: scale={self.relevance_bias_scale}')

        if self.use_mean_pool and not getattr(cfg, 'pure_inference', False):
            self.mean_pool_dropout = nn.Dropout(p=cfg.mean_pool_dropout)

            if self.use_proj_head:
                proj_hidden = self.hidden_size * self.proj_hidden_ratio
                self.proj_head = ContrastProjectionHead(self.hidden_size, proj_hidden)
                print(
                    f'Projection head initialized: '
                    f'{self.hidden_size} -> {proj_hidden} -> {self.hidden_size} '
                    f'(ratio={self.proj_hidden_ratio}, applied token-wise before mean pooling)'
                )
        elif self.use_mean_pool:
            print('Skipping mean-pool contrastive module initialization (pure inference mode)')

    def _create_decoder(self, cfg: CoMemConfig) -> AutoModelForCausalLM:
        """Create and configure the decoder model."""
        if not torch.cuda.is_available():
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=True,
                device_map=cfg.device_map
            )

        if cfg.quantization == "no":
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flex_attention",#"flex_attention", #cfg.attn_implementation,
                device_map=cfg.device_map
            )
        elif cfg.quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                quantization_config=quant_config,
                attn_implementation=cfg.attn_implementation,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=True,
                device_map=cfg.device_map
            )
        elif cfg.quantization == "int8":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype='bfloat16',
            )
            return AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                quantization_config=quant_config,
                attn_implementation=cfg.attn_implementation,
                torch_dtype=torch.bfloat16,
                resume_download=True,
                trust_remote_code=True,
                device_map=cfg.device_map
            )
        else:
            raise NotImplementedError(f"Quantization {cfg.quantization} not supported")

    def _setup_lora_adapters(self, cfg: CoMemConfig):
        """Setup LoRA adapters based on training stage."""
        peft_config = self._get_peft_config(lora_r=cfg.lora_r)

        if cfg.training_stage == "stage1" and cfg.load_adapters:
            print('Loading encoder and decoder adapter for stage1')
            self.decoder.add_adapter(peft_config, 'decoder_adapter')
            self.adapter_keys.append('decoder_adapter')
            self.decoder.add_adapter(peft_config, 'encoder_adapter')
            self.adapter_keys.append('encoder_adapter')
        elif cfg.training_stage == 'stage1_2':
            if not cfg.load_adapters:
                print('Loading decoder adapter for stage1_2')
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')
            elif cfg.load_adapters:
                print('Loading encoder and decoder adapter for stage1_2')
                self.decoder.add_adapter(peft_config, 'encoder_adapter')
                self.adapter_keys.append('encoder_adapter')
                self.decoder.add_adapter(peft_config, 'decoder_adapter')
                self.adapter_keys.append('decoder_adapter')

    def _setup_adapter_training(self):
        """Setup adapters for training."""
        for adapter_key in self.adapter_keys:
            self._set_active_adapters(adapter_key)
            print(f'Adapter {adapter_key} trainable parameters: {self.num_parameters(only_trainable=True)}')
        self._set_all_adapters()

    def _adapter_should_train(self, adapter_key: str) -> bool:
        """Return whether a named LoRA adapter should be updated by the optimizer."""
        if adapter_key == 'encoder_adapter':
            return self.train_encoder_adapter
        if adapter_key == 'decoder_adapter':
            return self.train_decoder_adapter
        return True

    def _get_adapter_params(self, adapter_key: str):
        """Cache parameters that belong to a specific PEFT adapter."""
        if adapter_key not in self._adapter_param_cache:
            self._adapter_param_cache[adapter_key] = [
                (name, param)
                for name, param in self.decoder.named_parameters()
                if adapter_key in name
            ]
        return self._adapter_param_cache[adapter_key]

    def _enforce_adapter_trainability(self, verbose: bool = False):
        """Apply encoder/decoder adapter trainability flags after PEFT adapter switches."""
        if not self.lora:
            return

        for adapter_key in self.adapter_keys:
            should_train = self._adapter_should_train(adapter_key)
            adapter_params = self._get_adapter_params(adapter_key)
            for _, param in adapter_params:
                param.requires_grad = should_train

            if verbose:
                trainable = sum(param.numel() for _, param in adapter_params if param.requires_grad)
                total = sum(param.numel() for _, param in adapter_params)
                print(
                    f'Adapter {adapter_key}: trainable={should_train}, '
                    f'trainable parameters={trainable}, total parameters={total}'
                )

    def _set_active_adapters(self, adapters):
        """Switch active PEFT adapters and re-apply per-adapter freezing."""
        self.decoder.set_adapter(adapters)
        self._enforce_adapter_trainability()

    def _configure_generation_config(self):
        """Configure generation parameters."""
        self.decoder.generation_config.top_p = None
        self.decoder.generation_config.temperature = None
        self.decoder.generation_config.pad_token_id = self.decoder_tokenizer.pad_token_id

    @staticmethod
    def _create_decoder_tokenizer(cfg: CoMemConfig) -> AutoTokenizer:
        """Create and configure the decoder tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.decoder_model_name,
            use_fast=True,
            padding_side='left'
        )

        # Define special tokens
        n_mem_tokens = cfg.doc_max_length // cfg.compr_rate
        existing_special_tokens = tokenizer.special_tokens_map.get("additional_special_tokens", [])

        if cfg.different_mem_tokens:
            mem_tokens = [f'<MEM{i}>' for i in range(n_mem_tokens)]
            tokenizer.add_special_tokens({
                'additional_special_tokens': existing_special_tokens + mem_tokens + ['<AE>', '<ENC>', '<SEP>','<QUERY>'] # add query 分割符？
            })
            tokenizer.mem_tokens = mem_tokens
        else:
            tokenizer.add_special_tokens({
                'additional_special_tokens': existing_special_tokens + ['<MEM>', '<AE>', '<ENC>', '<SEP>', '<QUERY>']
            })
            tokenizer.mem_tokens = ['<MEM>'] * n_mem_tokens

        tokenizer.mem_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenizer.mem_tokens]
        tokenizer.mem_token_ids_pt = torch.LongTensor(tokenizer.mem_token_ids)

        # Additional special tokens
        tokenizer.ae_token = '<AE>'
        tokenizer.ae_token_id = tokenizer.convert_tokens_to_ids('<AE>')
        tokenizer.enc_token = '<ENC>'
        tokenizer.sep_token = '<SEP>'
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids('<SEP>')
        tokenizer.query_token = '<QUERY>'
        tokenizer.query_token_id = tokenizer.convert_tokens_to_ids('<QUERY>')

        # Handle model-specific tokens
        if tokenizer.bos_token is None and 'qwen' in cfg.decoder_model_name.lower():
            tokenizer.bos_token = tokenizer.special_tokens_map['additional_special_tokens'][0]
            tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)

        if tokenizer.eos_token is None and "qwen" in cfg.decoder_model_name.lower():
            tokenizer.eos_token = tokenizer.special_tokens_map['additional_special_tokens'][1]
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        # KBTC training tokens
        if cfg.kbtc_training:
            tokenizer.add_special_tokens({'additional_special_tokens': ['<KBTC>']})
            tokenizer.kbtc_token = '<KBTC>'
            tokenizer.kbtc_token_id = tokenizer.convert_tokens_to_ids('<KBTC>')

        # Set pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.bos_token_id

        print(f'Memory token count: {n_mem_tokens}')
        return tokenizer

    def _get_peft_config(self, lora_r: int) -> LoraConfig:
        """Build the PEFT configuration."""
        return LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_r,
            lora_alpha=2*lora_r,
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"
                ], #'all-linear',
            lora_dropout=0.1
        )

    def _prepare_mem_tokens_optimization(self):
        """Setup memory token optimization if enabled."""
        if self.config.optimize_mem_tokens and self.compr is None:
            # Enable gradients for input embeddings
            self.decoder.get_input_embeddings().weight.requires_grad = True

            # Apply hook to zero gradients except for memory tokens
            def hook(grad):
                mask = torch.zeros_like(grad)
                mask[self.decoder_tokenizer.mem_token_ids] = 1.0
                return grad * mask

            self.decoder.get_input_embeddings().weight.register_hook(hook)

    def _set_all_adapters(self):
        """Activate all adapters for training."""
        if len(self.adapter_keys) > 0:
            self._set_active_adapters(self.adapter_keys)

    # Core compression and generation methods
    def compress(self, enc_input_ids: torch.Tensor, enc_attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress input documents."""
        if self.compr:
            return self.compr(enc_input_ids, enc_attention_mask)
        else:
            return self._compr_decoder(enc_input_ids, enc_attention_mask)

    def _compr_decoder(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Use decoder as compressor."""
        assert input_ids.size() == attention_mask.size()

        if 'encoder_adapter' in self.adapter_keys:
            self._set_active_adapters('encoder_adapter')
        else:
            raise ValueError(f"encoder_adapter not in adapter_keys: {self.adapter_keys}")

        block_mask = create_encoder_block_mask(attention_mask, self.n_mem_tokens, input_ids.device)

        # Inject query relevance bias into FlexAttention score_mod
        if self.use_query_relevance_bias:
            rel_bias = self._compute_query_relevance_bias(input_ids, attention_mask)
            if rel_bias is not None:
                set_query_relevance_bias(rel_bias, scale=self.relevance_bias_scale)

        # Get embeddings from decoder
        try:
            emb = self.decoder(
                input_ids=input_ids,
                attention_mask=block_mask,
                output_hidden_states=True
            ).hidden_states[-1]
        finally:
            if self.use_query_relevance_bias:
                clear_query_relevance_bias()

        # Create mask for memory tokens
        mask = torch.isin(
            input_ids,
            self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device)
        )

        mse_loss = torch.tensor(0.0, device=emb.device)

        return emb[mask].reshape(emb.size(0), -1, emb.size(-1)), mse_loss

    def _get_query_context_masks(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build masks for query tokens and document-context tokens.

        Query tokens are extracted from between the two ``<QUERY>`` markers.
        Context tokens are extracted after the second ``<QUERY>`` marker while
        excluding appended memory slots and special tokens such as ``<EOS>``.
        """
        query_token_id = self.decoder_tokenizer.query_token_id
        mem_token_ids = self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device)
        special_token_ids = torch.tensor(
            self.decoder_tokenizer.all_special_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )

        is_query_marker = input_ids == query_token_id
        is_mem_token = torch.isin(input_ids, mem_token_ids)
        is_special_token = torch.isin(input_ids, special_token_ids)
        valid_tokens = attention_mask > 0
        marker_cumsum = is_query_marker.cumsum(dim=1)
        n_markers = is_query_marker.sum(dim=1)

        is_query_content = (
            (marker_cumsum == 1) & valid_tokens & ~is_query_marker & ~is_special_token
        )
        is_context = (
            (marker_cumsum >= 2) & valid_tokens & ~is_special_token & ~is_mem_token
        )
        return is_query_content, is_context, n_markers

    def _compute_query_relevance_bias(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute per-position query relevance bias from initial token embeddings.

        For each sample, identifies query content (between <QUERY> markers) and
        context content (after the second <QUERY> marker and before appended
        memory tokens). Returns a [B, S] float32 tensor of cosine similarities
        between each context-position embedding and the mean query embedding.
        Non-context positions are zeroed. Float32 is required because
        flex_attention score_mod always operates in float32.
        """
        # Cast to float32 for stable cosine similarity and compatibility
        # with flex_attention score_mod (always float32).
        # Gradients flow back through the bias to the embedding table,
        # providing an additional training signal that encourages
        # query-relevant context tokens to be more distinguishable.
        embeddings = self.decoder.get_input_embeddings()(input_ids).float()  # [B, S, H]

        is_query_content, is_context, n_markers = self._get_query_context_masks(
            input_ids, attention_mask
        )
        if (n_markers < 2).all():
            return None

        query_mask = is_query_content.unsqueeze(-1).float()          # [B, S, 1]
        query_count = query_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
        mean_query_emb = (embeddings * query_mask).sum(dim=1, keepdim=True) / query_count  # [B, 1, H]

        # Normalise for cosine similarity (all float32 now)
        mean_query_emb = F.normalize(mean_query_emb, dim=-1)     # [B, 1, H]
        doc_emb_norm = F.normalize(embeddings, dim=-1)           # [B, S, H]

        # Cosine similarity of every position with the mean query embedding
        relevance = torch.bmm(doc_emb_norm, mean_query_emb.transpose(1, 2)).squeeze(-1)  # [B, S]
        relevance = relevance * is_context.float()

        # For samples without valid query markers, zero the whole row
        relevance = relevance * (n_markers >= 2).unsqueeze(-1).float()

        return relevance  # [B, S] float32

    def generate_from_paraphrase(self, questions: list[str], documents: list[list[str]], max_new_tokens: int = 128) -> list[str]:
        """
        Generates answers from documents (via compression then decoding)
        questions: list of string
        documents: list of list of strings (they should all be of equal length: the nb of doc for each question)
        """
        self.generation_top_k = len(documents[0])
        assert len(documents) == len(questions)
        assert all([len(context) == len(documents[0]) for context in documents])
        flat_documents = sum(documents, [])

        model_input = {}

        # Creating encoder inputs:
        input_encoder = self._prepare_encoder_inputs(flat_documents, max_length=self.doc_max_length)
        device = self.decoder.device
        model_input['enc_input_ids'], model_input['enc_attention_mask'] = input_encoder['input_ids'].to(device), input_encoder['attention_mask'].to(device)

        # Creating decoder inputs
        instr = [self._blend_prompt_and_memory_tokens(query="", stage = "stage1", paraphrase_loss = True) for q in questions]
        inp_dec = self.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True,  max_length=1024)
        model_input['dec_input_ids'], model_input['dec_attention_mask'] = inp_dec['input_ids'].to(device), inp_dec['attention_mask'].to(device)

        # Generation
        return self._generate(model_input, max_new_tokens=max_new_tokens)


    def generate_from_text(self,
                          questions: List[str],
                          documents: List[List[str]],
                          max_new_tokens: int = 128) -> List[str]:
        """Generate answers from documents via compression then decoding."""
        self.generation_top_k = len(documents[0])
        assert len(documents) == len(questions)
        assert all(len(context) == len(documents[0]) for context in documents)

        flat_documents = sum(documents, [])

        # Create encoder inputs
        input_encoder = self._prepare_encoder_inputs(flat_documents, max_length=self.doc_max_length)
        device = self.decoder.device
        enc_input_ids = input_encoder['input_ids'].to(device)
        enc_attention_mask = input_encoder['attention_mask'].to(device)

        # Create decoder inputs
        instructions = [self._blend_prompt_and_memory_tokens(query=q, stage="stage1_2") for q in questions]
        inp_dec = self.decoder_tokenizer(
            instructions,
            return_tensors='pt',
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=1024
        )
        dec_input_ids = inp_dec['input_ids'].to(device)
        dec_attention_mask = inp_dec['attention_mask'].to(device)

        # Generate
        return self._generate({
            'enc_input_ids': enc_input_ids,
            'enc_attention_mask': enc_attention_mask,
            'dec_input_ids': dec_input_ids,
            'dec_attention_mask': dec_attention_mask
        }, max_new_tokens=max_new_tokens)

    def generate_from_compressed_documents_and_questions(self,
                                                        questions: List[str],
                                                        compressed_documents: torch.Tensor,
                                                        max_new_tokens: int = 128) -> List[str]:
        """Generate answers from compressed documents."""
        self.generation_top_k = compressed_documents.size(0) // len(questions)
        assert compressed_documents.size(0) % self.generation_top_k == 0

        # Create decoder inputs
        instructions = [self._blend_prompt_and_memory_tokens(query=q, stage="stage1_2") for q in questions]
        inp_dec = self.decoder_tokenizer(
            instructions,
            return_tensors='pt',
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=1024
        )
        device = self.decoder.device
        dec_input_ids = inp_dec['input_ids'].to(device)
        dec_attention_mask = inp_dec['attention_mask'].to(device)

        # Create input decoder embeddings from prompt + compressed documents
        inputs_embeds = self._replace_emb(compressed_documents, dec_input_ids)

        # Activate decoder generator
        if 'decoder_adapter' in self.adapter_keys:
            self._set_active_adapters('decoder_adapter')

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_attention_mask,
            max_new_tokens=max_new_tokens
        )

        return self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def compress_documents(self, documents: List[str], q_texts=None) -> torch.Tensor: # change add q_text
        """Compress a list of documents."""
        input_encoder = self._prepare_encoder_inputs(documents, max_length=self.doc_max_length,q_texts=q_texts) ## change add q_text
        enc_input_ids = input_encoder['input_ids'].to(self.decoder.device)
        attention_mask = input_encoder['attention_mask'].to(self.decoder.device)
        return self.compress(enc_input_ids=enc_input_ids, enc_attention_mask=attention_mask)

    # Helper methods
    def _prepare_encoder_inputs(self, texts: List[str], max_length: int, q_texts: List[str] = None) -> Dict[str, torch.Tensor]:
        """Create inputs for the encoder."""
        if q_texts is not None:
            assert len(texts) == len(q_texts)

        if self.compr is None:
            return self._prepare_encoder_inputs_to_decoder(texts, max_length, q_texts)
        else:
            return self.compr.prepare_inputs(texts, max_length, q_texts)

    def _prepare_encoder_inputs_to_decoder(self, texts: List[str], max_length: int, q_texts: List[str] = None) -> Dict[str, torch.Tensor]:
        """Prepare encoder inputs when using decoder as compressor."""
        normal_query = "Extract potentially useful information from the text to help answer the question."
        if q_texts is not None:
            texts_to_encode = [
                self.decoder_tokenizer.enc_token +
                self.decoder_tokenizer.bos_token +
                # '\nQuery:\n' + query +
                self.decoder_tokenizer.query_token + '\nQuery:\n' + query + self.decoder_tokenizer.query_token +
                'Document:\n' + text +
                self.decoder_tokenizer.eos_token
                for text, query in zip(texts, q_texts)
            ]
            inp_enc = self.decoder_tokenizer(
                texts_to_encode,
                return_tensors='pt',
                padding='max_length',
                max_length=max_length + 8,
                truncation=True,
                add_special_tokens=False
            )
        else:
            inp_enc = [
                self.decoder_tokenizer.enc_token +
                self.decoder_tokenizer.bos_token +
                text +
                self.decoder_tokenizer.eos_token
                for text in texts
            ]
            inp_enc = self.decoder_tokenizer(
                inp_enc,
                return_tensors='pt',
                padding="max_length",
                max_length=max_length + 3,
                truncation=True,
                add_special_tokens=False
            )

        num_mem_tokens = self.doc_max_length // self.compr_rate
        assert num_mem_tokens == len(self.decoder_tokenizer.mem_tokens)

        inp_enc['input_ids'], inp_enc['attention_mask'] = add_memory_tokens_to_inputs(
            inp_enc['input_ids'],
            inp_enc['attention_mask'],
            num_mem_tokens,
            tokenizer=self.decoder_tokenizer
        )

        return inp_enc

    def _replace_emb(self, compressed_embs: torch.Tensor, dec_input_ids: torch.Tensor) -> torch.Tensor:
        """Replace memory tokens in decoder input with compressed embeddings."""
        indices = range(0, compressed_embs.size(0) + 1, self.generation_top_k)
        return self._replace_embeddings(compressed_embs, dec_input_ids, indices)

    def _replace_embeddings(self, compressed_embs: torch.Tensor, dec_input_ids: torch.Tensor, indices: range) -> torch.Tensor:
        """Replace memory tokens with compressed embeddings."""
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        num_embs = compressed_embs.size(1)
        slot_len = num_embs + (1 if self.sep else 0)

        # Get first memory token indices
        first_mem_token_indices = torch.argmax(
            (dec_input_ids == self.decoder_tokenizer.mem_token_ids[0]).int(), dim=1
        )
        batch_size = inputs_embeds.size(0)

        # Replace with compressed embeddings
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = first_mem_token_indices[i].item() + (j - indices[i]) * slot_len
                assert inputs_embeds[i, start_idx:start_idx + num_embs, :].size() == compressed_embs[j].size()
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]

        return inputs_embeds

    def _blend_prompt_and_memory_tokens(self, query: str, answer: str = None, qa_loss: bool = False,
                                       paraphrase_loss: bool = False, stage: str = "stage1") -> Tuple[int, str]:
        """Blend prompt with memory tokens for different training stages."""
        mem_tokens_str = ''.join(self.decoder_tokenizer.mem_tokens) + self.decoder_tokenizer.sep_token
        docs = mem_tokens_str * self.generation_top_k

        if stage == "stage1":
            if qa_loss:
                sft_query, sft_answer = self._normalize_sft_qa_pair(query, answer)
                return self._blend_standard_prompt(docs, sft_query, sft_answer)
            elif paraphrase_loss:
                return self._blend_paraphrase_prompt(docs, answer)
        elif stage == "stage1_2":
            return self._blend_standard_prompt(docs, query, answer)

        raise ValueError(f"Unknown stage: {stage}")

    @staticmethod
    def _normalize_sft_qa_pair(query, answer) -> Tuple[str, str]:
        """Normalize stage1 QA samples to a single query-answer pair for SFT."""
        if isinstance(query, list):
            query = query[0] if query else ""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        return str(query), str(answer)

    def _blend_qa_prompt(self, docs: str, query: List[str], answer: List[str]) -> Tuple[int, str]:
        """Create QA prompt for stage 1."""
        prompt_system = 'You are a helpful assistant. Given a document, your task is to generate some single questions to cover all key information of the document and answer them sequentially.'
        prompt_user = f"Background:\n{docs}"

        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]

        qa_lines = [f"Question: {q}\nAnswer: {a}" for q, a in zip(query, answer)]
        query_answer = "\n".join(qa_lines)
        assistant_prompt = [{"role": "assistant", "content": query_answer}]

        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            response = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt + assistant_prompt,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
        except TemplateError as e:
            if "System role not supported" in str(e):
                messages = [{"role": "user", "content": sys_prompt[0]['content'] + '\n' + user_prompt[0]['content']}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                # Handle response for unsupported system role
                messages_with_answer = messages + assistant_prompt
                response = self.decoder_tokenizer.apply_chat_template(
                    messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                raise e

        return prompt_len, response

    def _blend_paraphrase_prompt(self, docs: str, answer: str) -> Tuple[int, str]:
        """Create paraphrase prompt for stage 1."""
        prompt_system = 'You are a helpful assistant. Your task is follow the instructions to paraphrase the background information.'
        prompt_user = random.choice(PARAPHRASE_INSTRUCTIONS).format(docs=docs)

        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]

        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            if answer is None:
                return prompt

            assistant_prompt = [{"role": "assistant", "content": answer}]
            response = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt + assistant_prompt,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
        except TemplateError as e:
            if "System role not supported" in str(e):
                combined_content = prompt_system + '\n' + prompt_user.replace(':\ ', ': ')
                messages = [{"role": "user", "content": combined_content}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                if answer is None:
                    return prompt
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                messages_with_answer = messages + [{"role": "assistant", "content": answer}]
                response = self.decoder_tokenizer.apply_chat_template(
                    messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                raise e

        return prompt_len, response

    def _blend_standard_prompt(self, docs: str, query: str, answer: str) -> Tuple[int, str]:
        """Create standard prompt for stage 1_2."""
        prompt_system = 'You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions.'
        prompt_user = f"Background:\n{docs}\n\nQuestion:{query}"

        sys_prompt = [{"role": "system", "content": prompt_system}]
        user_prompt = [{"role": "user", "content": prompt_user.replace(':\ ', ': ')}]

        try:
            prompt = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            if answer is None:
                return prompt

            assistant_prompt = [{"role": "assistant", "content": answer}]
            response = self.decoder_tokenizer.apply_chat_template(
                sys_prompt + user_prompt + assistant_prompt,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False
            )
            prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
        except TemplateError as e:
            if "System role not supported" in str(e):
                combined_content = prompt_system + '\n' + prompt_user.replace(':\ ', ': ')
                messages = [{"role": "user", "content": combined_content}]
                prompt = self.decoder_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                if answer is None:
                    return prompt
                prompt_len = len(self.decoder_tokenizer.encode(prompt, add_special_tokens=False))
                messages_with_answer = messages + [{"role": "assistant", "content": answer}]
                response = self.decoder_tokenizer.apply_chat_template(
                    messages_with_answer, tokenize=False, add_generation_prompt=False, enable_thinking=False
                )
            else:
                raise e

        return prompt_len, response

    # Model saving and loading methods
    def save_pretrained(self, save_directory: str, **kwargs):
        """Save only the LoRA adapters and their configurations."""
        if self.lora:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            # Save LoRA adapter weights
            torch.save(
                self._get_all_adapters_state_dict(),
                os.path.join(save_directory, "adapters.pth")
            )

            # Save first and last layers of decoder
            torch.save(
                self._get_decoder_first_and_last_layer_state_dict(),
                os.path.join(save_directory, "decoder_first_last_layers.pth")
            )

            # Save contrastive projection head used by mean-pool training.
            if self.use_mean_pool and self.use_proj_head and hasattr(self, 'proj_head'):
                torch.save(
                    {'proj_head': self.proj_head.state_dict()},
                    os.path.join(save_directory, "proj_heads.pth")
                )

            # Save configuration
            self.config.save_pretrained(save_directory)
        else:
            super().save_pretrained(save_directory, **kwargs)

    def _get_all_adapters_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Return the state dicts of all adapters."""
        return {
            key: {k: v.cpu() for k, v in self.decoder.get_adapter_state_dict(key).items()}
            for key in self.adapter_keys
        }

    def _get_decoder_first_and_last_layer_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get first and last layers that change when adding tokens."""
        out = {}
        for k, v in self.decoder.named_parameters():
            if 'lm_head.weight' in k or 'embed_tokens.weight' in k:
                out[k] = v.cpu()
        return out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """Load model from pretrained checkpoint."""
        checkpoint_subfolder = kwargs.pop("subfolder", None)

        def checkpoint_file(filename: str) -> str:
            """Resolve a checkpoint file locally or from a Hub subfolder."""
            if checkpoint_subfolder:
                local_path = os.path.join(
                    pretrained_model_name_or_path, checkpoint_subfolder, filename
                )
            else:
                local_path = os.path.join(pretrained_model_name_or_path, filename)
            try:
                return hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=filename,
                    subfolder=checkpoint_subfolder,
                )
            except Exception:
                return local_path

        # Load configuration
        config = CoMemConfig.from_pretrained(
            pretrained_model_name_or_path, subfolder=checkpoint_subfolder
        )

        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        map_location = torch.device("cpu") if not torch.cuda.is_available() else None

        if config.lora:
            # Delay adapter construction
            config.load_adapters = False
            config.delay_adapter_init = True
            if 'device_map' in kwargs:
                config.device_map = kwargs['device_map']

            # Initialize model
            print(f"Initializing model from trained checkpoint: {config}")
            model = cls(config)

            # Load first and last layers
            first_and_last_layers_path = checkpoint_file("decoder_first_last_layers.pth")

            if os.path.exists(first_and_last_layers_path):
                first_and_last_decoder_state_dict = torch.load(
                    first_and_last_layers_path, map_location=map_location, weights_only=True
                )
                for key in first_and_last_decoder_state_dict:
                    assert key in model.decoder.state_dict()
                model.decoder.load_state_dict(first_and_last_decoder_state_dict, strict=False)
            else:
                print(f'First and last layer not found: {first_and_last_layers_path}')

            peft_config = model._get_peft_config(lora_r=config.lora_r)

            # Load LoRA adapters
            adapters_path = checkpoint_file("adapters.pth")

            if os.path.exists(adapters_path):
                adapters_state_dict = torch.load(adapters_path, map_location=map_location, weights_only=True)
                model._load_adapters_from_state_dict(adapters_state_dict, peft_config, config)
                print(f'Loaded checkpoint adapters: {model.adapter_keys}')
            else:
                warnings.warn(f'Adapters not found at {adapters_path}')

            model._set_all_adapters()
            model._enforce_adapter_trainability(verbose=True)
            config.load_adapters = True
            config.delay_adapter_init = False

            # Load projection head weights used by mean-pool contrastive training.
            if (model.use_mean_pool
                    and getattr(model, 'use_proj_head', False)
                    and hasattr(model, 'proj_head')
                    and not getattr(config, 'pure_inference', False)):
                proj_heads_path = checkpoint_file("proj_heads.pth")

                if os.path.exists(proj_heads_path):
                    proj_heads_state = torch.load(
                        proj_heads_path, map_location=map_location, weights_only=True
                    )
                    model.proj_head.load_state_dict(proj_heads_state['proj_head'])
                    print(f'Loaded projection head from {proj_heads_path}')
                else:
                    print(f'Projection head checkpoint not found at {proj_heads_path}, '
                          f'using random initialization')

            return model
        else:
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def _load_adapters_from_state_dict(self, adapters_state_dict: Dict, peft_config: LoraConfig, config: CoMemConfig):
        """Load adapters from state dict based on training stage."""
        if not getattr(config, 'pure_inference', False):
            for key, val in adapters_state_dict.items():
                self._load_adapter_from_state_dict(
                    peft_config=peft_config,
                    adapter_name=key,
                    adapter_state_dict=val
                )
        else:
            # Load all adapters for pure inference
            for key, val in adapters_state_dict.items():
                self._load_adapter_from_state_dict(
                    peft_config=peft_config,
                    adapter_name=key,
                    adapter_state_dict=val
                )

    def _load_adapter_from_state_dict(self, peft_config: LoraConfig, adapter_name: str, adapter_state_dict: Dict):
        """Create adapter from state dict."""
        print(f'Loading checkpoint adapter: {adapter_name}')
        self.decoder.load_adapter(
            peft_config=peft_config,
            adapter_name=adapter_name,
            adapter_state_dict=adapter_state_dict
        )
        self.adapter_keys.append(adapter_name)
        self._adapter_param_cache.pop(adapter_name, None)

    # Forward pass methods
    def forward(self,
                batch: Dict = None,
                questions: List[str] = None,
                documents: List[List[str]] = None,
                answers: List[str] = None,
                original_answer_gen_api: str = None) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with support for both batch and legacy interfaces.

        Args:
            batch: Preprocessed batch dict (new interface)
            questions: List of questions (legacy interface)
            documents: List of document lists (legacy interface)
            answers: List of answers (legacy interface)
            original_answer_gen_api: API URL for generation (legacy interface)

        Returns:
            Tuple of (loss, additional_outputs_dict)
        """
        if batch is not None:
            return self._forward_batch(batch)
        else:
            return self._forward_legacy(questions, documents, answers, original_answer_gen_api)

    def _forward_batch(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Handle batch-based forward pass."""
        stage = batch.get("stage", None)

        if stage in ["stage1", "stage1_2"]:
            return self._forward_stage1_batch(batch)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _forward_stage1_batch(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for stage 1 training."""
        enc_input_ids = batch["enc_input_ids"].to(self.decoder.device)
        enc_attention_mask = batch["enc_attention_mask"].to(self.decoder.device)
        dec_input_ids = batch["dec_input_ids"].to(self.decoder.device)
        dec_attention_mask = batch["dec_attention_mask"].to(self.decoder.device)
        labels = batch["labels"].to(self.decoder.device)

        if self.use_mean_pool and self.training:
            out = self._forward_stage1_simcse_in_batch(
                enc_input_ids=enc_input_ids,
                enc_attention_mask=enc_attention_mask,
                dec_input_ids=dec_input_ids,
                dec_attention_mask=dec_attention_mask,
                labels=labels,
            )
        else:
            out = self._forward_stage_1(
                enc_input_ids=enc_input_ids,
                enc_attention_mask=enc_attention_mask,
                dec_input_ids=dec_input_ids,
                dec_attention_mask=dec_attention_mask,
                labels=labels,
            )
        return out["loss"], {
            "logits": out["logits"],
            "mse_loss": out["mse_loss"],
            "simcse_loss": out["simcse_loss"],
            "simcse_pos_logit": out["simcse_pos_logit"],
            "simcse_neg_logsumexp": out["simcse_neg_logsumexp"],
            "ortho_loss": out["ortho_loss"],
        }


    def _forward_stage_1(self,
                        enc_input_ids: torch.LongTensor = None,
                        enc_attention_mask: torch.LongTensor = None,
                        dec_input_ids: torch.LongTensor = None,
                        dec_attention_mask: torch.LongTensor = None,
                        labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """Stage 1 forward pass for document compression and QA."""
        assert enc_input_ids.size() == enc_attention_mask.size()

        # Flatten 3D inputs to 2D if needed
        if len(enc_input_ids.size()) == 3:
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)

        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k

        # Compress documents
        compressed_embs, mse_loss = self.compress(enc_input_ids, enc_attention_mask) # compress

        # Replace memory tokens with compressed embeddings
        inputs_embeds = self._replace_emb(compressed_embs, dec_input_ids)

        # Detach if compressor-only training
        if (self.training_form == "compressor") and (self.compr is None):
            inputs_embeds = inputs_embeds.detach()

        # Set decoder adapter
        if 'decoder_adapter' in self.adapter_keys:
            self._set_active_adapters('decoder_adapter')

        # Convert 2D attention mask to BlockMask for flex_attention
        dec_block_mask = create_decoder_block_mask(dec_attention_mask, dec_input_ids.device)

        # Forward through decoder
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_block_mask,
            labels=labels
        )

        # Reactivate all adapters
        self._set_all_adapters()

        return {
            "loss": decoder_outputs.loss,
            "logits": decoder_outputs.logits,
            "mse_loss": mse_loss,
            "simcse_loss": torch.tensor(0.0),
            "simcse_pos_logit": torch.tensor(0.0),
            "simcse_neg_logsumexp": torch.tensor(0.0),
            "ortho_loss": torch.tensor(0.0),
        }

    def _compute_simcse_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
        """
        Compute SimCSE contrastive loss (standard implementation).

        SimCSE uses the same sample with different dropout masks as positive pairs,
        and other samples in the batch as negative pairs.

        Reference: https://arxiv.org/abs/2104.08821

        Args:
            z1: First representation after dropout [batch_size, hidden_size]
            z2: Second representation after dropout [batch_size, hidden_size]
            temperature: Temperature for softmax scaling (default 0.05, i.e., scale by 20)

        Returns:
            SimCSE contrastive loss
        """
        batch_size = z1.size(0)

        # Interleave z1 and z2: [z1_0, z2_0, z1_1, z2_1, ...]
        # Shape: [2 * batch_size, hidden_size]
        batch_emb = torch.stack([z1, z2], dim=1).view(2 * batch_size, -1)

        # Normalize representations for cosine similarity
        norm_emb = F.normalize(batch_emb, p=2, dim=1)

        # Compute similarity matrix [2*batch_size, 2*batch_size]
        sim_score = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # Mask out diagonal (self-similarity) with large negative value
        sim_score = sim_score - torch.eye(2 * batch_size, device=z1.device) * 1e12

        # Scale by temperature (1/0.05 = 20)
        sim_score = sim_score / temperature

        # Construct labels: positive pairs are adjacent (0<->1, 2<->3, ...)
        # Label for position 0 is 1, for position 1 is 0, for position 2 is 3, etc.
        y_true = torch.cat([
            torch.arange(1, 2 * batch_size, step=2, dtype=torch.long, device=z1.device).unsqueeze(1),
            torch.arange(0, 2 * batch_size, step=2, dtype=torch.long, device=z1.device).unsqueeze(1)
        ], dim=1).reshape(2 * batch_size)

        # Cross entropy loss
        loss = F.cross_entropy(sim_score, y_true)

        return loss

    @torch.no_grad()
    def _concat_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Gather tensors from all GPUs using all_gather (no gradient).

        Args:
            tensor: Local tensor [batch_size, hidden_size]

        Returns:
            Gathered tensor [world_size * batch_size, hidden_size]
        """
        if not torch.distributed.is_initialized():
            return tensor

        world_size = torch.distributed.get_world_size()
        tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

        return torch.cat(tensors_gather, dim=0)

    def _compute_ortho_loss(self, compressed_embs: torch.Tensor) -> torch.Tensor:
        """
        Orthogonality regularization on per-sample mem tokens.

        For each sample, L2-normalizes its n_mem_tokens vectors, computes
        the Gram matrix (cosine similarities), and penalizes the squared
        off-diagonal entries:

            L = (1 / B) * sum_i || G_i - I ||_F^2 / (n * (n - 1))

        where G_i is the [n_mem, n_mem] cosine-similarity matrix for sample i.
        This pushes each mem token to encode distinct information.

        Args:
            compressed_embs: [B, n_mem_tokens, hidden_size]

        Returns:
            Scalar orthogonality loss.
        """
        B, n, H = compressed_embs.shape
        if n <= 1:
            return torch.tensor(0.0, device=compressed_embs.device)

        normed = F.normalize(compressed_embs.float(), dim=-1)       # [B, n, H]
        gram = torch.bmm(normed, normed.transpose(1, 2))            # [B, n, n]
        eye = torch.eye(n, device=gram.device, dtype=gram.dtype).unsqueeze(0)
        off_diag = gram - eye                                       # zero diagonal
        loss = (off_diag ** 2).sum() / (B * n * (n - 1))
        return loss

    # ------------------------------------------------------------------ #
    #  Mean-pool contrastive learning
    # ------------------------------------------------------------------ #

    def _encode_with_query_context_repr(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the encoder and return compressed mem, query, and context representations.

        The query representation is extracted from the encoder hidden states at
        positions between the two ``<QUERY>`` markers (causal self-attention
        within query tokens, so they represent the query only, not the document).
        The context representation is extracted after the second ``<QUERY>``
        marker, excluding appended memory tokens and other special tokens.

        Returns:
            compressed_embs: [B, n_mem_tokens, hidden_size]
            mse_loss:        scalar (currently 0.0)
            query_repr:      [B, hidden_size]  (mean-pooled query token hidden states)
            context_repr:    [B, hidden_size]  (mean-pooled context token hidden states)
        """
        assert input_ids.size() == attention_mask.size()

        if 'encoder_adapter' in self.adapter_keys:
            self._set_active_adapters('encoder_adapter')
        else:
            raise ValueError(f"encoder_adapter not in adapter_keys: {self.adapter_keys}")

        block_mask = create_encoder_block_mask(
            attention_mask, self.n_mem_tokens, input_ids.device
        )

        if self.use_query_relevance_bias:
            rel_bias = self._compute_query_relevance_bias(input_ids, attention_mask)
            if rel_bias is not None:
                set_query_relevance_bias(rel_bias, scale=self.relevance_bias_scale)

        try:
            emb = self.decoder(
                input_ids=input_ids,
                attention_mask=block_mask,
                output_hidden_states=True,
            ).hidden_states[-1]
        finally:
            if self.use_query_relevance_bias:
                clear_query_relevance_bias()

        # --- compressed mem embeddings ---
        mem_mask = torch.isin(
            input_ids,
            self.decoder_tokenizer.mem_token_ids_pt.to(input_ids.device),
        )
        compressed_embs = emb[mem_mask].reshape(emb.size(0), -1, emb.size(-1))

        is_query_content, is_context, _ = self._get_query_context_masks(
            input_ids, attention_mask
        )

        # --- query token representations ---
        query_mask_3d = is_query_content.unsqueeze(-1).float()           # [B, S, 1]
        query_count = query_mask_3d.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
        query_repr = (emb * query_mask_3d).sum(dim=1) / query_count.squeeze(-1)  # [B, H]

        # --- context token representations ---
        context_mask_3d = is_context.unsqueeze(-1).float()               # [B, S, 1]
        context_count = context_mask_3d.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1, 1]
        context_repr = (emb * context_mask_3d).sum(dim=1) / context_count.squeeze(-1)  # [B, H]

        mse_loss = torch.tensor(0.0, device=emb.device)
        return compressed_embs, mse_loss, query_repr, context_repr

    def _all_gather_with_grad(self, tensor: torch.Tensor) -> torch.Tensor:
        """all_gather that preserves gradient for the local rank's slice."""
        if not torch.distributed.is_initialized():
            return tensor
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor.detach())
        gathered[rank] = tensor  # keep gradient for local rank
        return torch.cat(gathered, dim=0)

    def _forward_stage1_simcse_in_batch(
        self,
        enc_input_ids: torch.LongTensor,
        enc_attention_mask: torch.LongTensor,
        dec_input_ids: torch.LongTensor,
        dec_attention_mask: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        """Stage 1 forward with mem-anchored contrastive loss.

        Positive pair: mem_i vs context_i.
        Negatives:     mem_i vs mem_j for all j != i.
        """
        assert enc_input_ids.size() == enc_attention_mask.size()

        if len(enc_input_ids.size()) == 3:
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)

        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k

        # Encode and extract both memory-token and context representations.
        compressed_embs, mse_loss, _, context_repr = self._encode_with_query_context_repr(
            enc_input_ids, enc_attention_mask
        )

        # ---------- contrastive loss ----------
        simcse_loss = torch.tensor(0.0, device=compressed_embs.device)
        simcse_pos_logit = torch.tensor(0.0, device=compressed_embs.device)
        simcse_neg_logsumexp = torch.tensor(0.0, device=compressed_embs.device)
        ortho_loss = torch.tensor(0.0, device=compressed_embs.device)
        if self.training:
            if self.use_proj_head:
                projected = self.proj_head(compressed_embs)
            else:
                projected = compressed_embs
            mem_repr = F.normalize(
                self.mean_pool_dropout(projected.mean(dim=1)), dim=1
            )  # [local_B, H]
            context_repr_norm = F.normalize(context_repr.detach(), dim=1)  # [local_B, H]

            all_mem = self._all_gather_with_grad(mem_repr)        # [global_B, H]
            all_context = self._concat_all_gather(context_repr_norm)   # [global_B, H]

            all_mem = all_mem.float()
            all_context = all_context.float()

            # For each anchor mem_i, use its paired context_i as the positive
            # score and all other mem_j as negatives.
            sim = all_mem @ all_mem.T / self.simcse_temperature  # [N, N]
            pos_sim = (all_mem * all_context).sum(dim=1) / self.simcse_temperature  # [N]
            sim.fill_diagonal_(0.0)
            sim[torch.arange(sim.size(0), device=sim.device), torch.arange(sim.size(0), device=sim.device)] = pos_sim
            labels_cl = torch.arange(sim.size(0), device=sim.device)
            simcse_loss = F.cross_entropy(sim, labels_cl)
            with torch.no_grad():
                simcse_pos_logit = pos_sim.mean()
                if sim.size(0) > 1:
                    neg_sim = all_mem.detach() @ all_mem.detach().T / self.simcse_temperature
                    neg_sim.fill_diagonal_(float("-inf"))
                    simcse_neg_logsumexp = torch.logsumexp(neg_sim, dim=1).mean()
            # Orthogonality regularization on mem tokens
            if self.ortho_loss_weight > 0:
                ortho_loss = self._compute_ortho_loss(compressed_embs)

        # ---------- decoder forward (LM loss) ----------
        inputs_embeds = self._replace_emb(compressed_embs, dec_input_ids)

        if (self.training_form == "compressor") and (self.compr is None):
            inputs_embeds = inputs_embeds.detach()

        if 'decoder_adapter' in self.adapter_keys:
            self._set_active_adapters('decoder_adapter')

        dec_block_mask = create_decoder_block_mask(dec_attention_mask, dec_input_ids.device)

        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=dec_block_mask,
            labels=labels,
        )

        self._set_all_adapters()

        return {
            "loss": decoder_outputs.loss,
            "logits": decoder_outputs.logits,
            "mse_loss": mse_loss,
            "simcse_loss": simcse_loss,
            "simcse_pos_logit": simcse_pos_logit,
            "simcse_neg_logsumexp": simcse_neg_logsumexp,
            "ortho_loss": ortho_loss,
        }

    def _replace_reasoning_embeddings(self,
                                    compressed_embs: torch.Tensor,
                                    dec_input_ids: torch.LongTensor,
                                    docs_per_example: List[int]) -> torch.Tensor:
        """Replace memory slots with compressed embeddings for reasoning."""
        device = dec_input_ids.device
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)

        num_embs = compressed_embs.size(1)
        slot_len = num_embs + (1 if getattr(self, "sep", False) else 0)

        if not isinstance(docs_per_example, torch.Tensor):
            docs_per_example = torch.tensor(docs_per_example, device=device, dtype=torch.long)
        else:
            docs_per_example = docs_per_example.to(device=device, dtype=torch.long)

        offsets = torch.zeros(docs_per_example.size(0) + 1, device=device, dtype=torch.long)
        offsets[1:] = torch.cumsum(docs_per_example, dim=0)
        total_docs = int(offsets[-1].item())
        assert total_docs == compressed_embs.size(0)

        mem_id = self.decoder_tokenizer.mem_token_ids[0]
        B, L, H = inputs_embeds.size()

        for i in range(B):
            # Find first memory token position
            mem_pos = (dec_input_ids[i] == mem_id).nonzero(as_tuple=True)[0]
            if mem_pos.numel() == 0:
                continue
            first_mem_idx = int(mem_pos[0].item())

            n_docs_i = int(docs_per_example[i].item())
            base = int(offsets[i].item())

            needed_len = first_mem_idx + n_docs_i * slot_len
            assert needed_len <= L

            for local_j in range(n_docs_i):
                global_j = base + local_j
                start_idx = first_mem_idx + local_j * slot_len
                target_slice = inputs_embeds[i, start_idx:start_idx + num_embs, :]
                src = compressed_embs[global_j]
                assert target_slice.size() == src.size()
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = src

        return inputs_embeds

    def _generate(self, model_input: Dict[str, torch.Tensor], max_new_tokens: int = 128,
                 return_doc_embeddings: bool = False) -> List[str]:
        """Generate text from model inputs."""
        enc_input_ids = model_input['enc_input_ids']
        enc_attention_mask = model_input['enc_attention_mask']
        dec_input_ids = model_input['dec_input_ids']
        dec_attention_mask = model_input['dec_attention_mask']

        assert enc_input_ids.size() == enc_attention_mask.size()

        if len(enc_input_ids.size()) == 3:
            batch_size, top_k, seq_length = enc_input_ids.size()
            enc_input_ids = enc_input_ids.view(batch_size * top_k, seq_length)
            enc_attention_mask = enc_attention_mask.view(batch_size * top_k, seq_length)

        assert enc_input_ids.size(0) == dec_input_ids.size(0) * self.generation_top_k

        compressed_embs, _ = self.compress(enc_input_ids.to('cuda'), enc_attention_mask.to('cuda'))
        inputs_embeds = self._replace_emb(compressed_embs, dec_input_ids.to('cuda'))

        if 'decoder_adapter' in self.adapter_keys:
            self._set_active_adapters('decoder_adapter')

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"),
            attention_mask=dec_attention_mask.to("cuda"),
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens
        )

        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        if return_doc_embeddings:
            assert 'batch_size' in locals() and 'top_k' in locals()
            compressed_embs = compressed_embs.view(batch_size, top_k, compressed_embs.size(1), compressed_embs.size(2))
            return decoded, compressed_embs
        else:
            return decoded


# Example usage and testing
if __name__ == '__main__':
    # Example configuration
    cfg = CoMemConfig(
        decoder_model_name='/mnt/ceph_rbd/model/Mistral-7B-Instruct-v0.2',
        compr_model_name="mistral_trimmed",
        compr_rate=64,
        compr_n_layers=5,
        compr_mlp_hidden_dim=8096,
        compr_use_mlp=False,
        lora=True,
        lora_compressor=True,
        training_form="both",
        load_adapters=True,
        kbtc_training=False,
        optimize_mem_tokens=True,
        different_mem_tokens=True,
        attn_implementation='flash_attention_2'
    )

    # Initialize model
    comem = CoMem(cfg)

    # Save and reload test
    comem.save_pretrained('test_ckpt')

    del comem
    torch.cuda.empty_cache()
    gc.collect()

    # Reload model
    comem = CoMem.from_pretrained('test_ckpt')
    print("Model successfully loaded!")
