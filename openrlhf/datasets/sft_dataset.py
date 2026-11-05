#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""
CLaRa Dataset and Collate Functions

This module provides dataset handling and batch collation for CLaRa training.
"""

import torch
from typing import Callable, List, Tuple, Dict, Any, Optional
from torch.utils.data import Dataset
from openrlhf.utils.utils import zero_pad_sequences


_FLEX_BLOCK_SIZE = 128

def _pad_to_block_multiple(input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_token_id: int):
    """Pad sequence length to the nearest multiple of _FLEX_BLOCK_SIZE.

    This avoids repeated torch.compile recompilations inside create_block_mask /
    flex_attention when padding="longest" produces many different lengths.
    """
    B, S = input_ids.shape
    remainder = S % _FLEX_BLOCK_SIZE
    if remainder == 0:
        return input_ids, attention_mask
    pad_len = _FLEX_BLOCK_SIZE - remainder
    input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_token_id)
    attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
    return input_ids, attention_mask


def make_collate_fn(clara_model, enc_max_len: int = 256, dec_max_len: int = 1024, qa_loss: bool = False):
    """
    Create a collate function for CLaRa training.
    
    Args:
        clara_model: CLaRa model instance
        enc_max_len: Maximum encoder sequence length
        dec_max_len: Maximum decoder sequence length
        qa_loss: Whether to use QA loss for joint training
        
    Returns:
        Collate function that processes batches for training
    """
    tokenizer = clara_model.decoder_tokenizer
    generation_top_k = clara_model.generation_top_k
    
    def _mask_prompt(labels: torch.Tensor, 
                     attention_mask: torch.Tensor, 
                     prompt_lengths: List[int], 
                     pad_token_id: int) -> torch.Tensor:
        """Mask prompt tokens in labels to only compute loss on answer tokens."""
        for i, prompt_len in enumerate(prompt_lengths):
            attn = attention_mask[i]
            valid_positions = attn.nonzero(as_tuple=True)[0]
            
            if len(valid_positions) == 0:
                continue
                
            first_valid = valid_positions[0].item()
            last_valid_plus1 = valid_positions[-1].item() + 1
            end_pos = min(first_valid + prompt_len, last_valid_plus1)
            labels[i, :end_pos] = -100
            
        return labels
    
    def _normalize_stage1_query(question, data_type: str) -> str:
        """Normalize stage1 query so encoder/decoder always reuse the same value."""
        # Paraphrase follows the current encoder behavior for an empty query.
        if data_type == "paraphrase":
            return ""

        if isinstance(question, list):
            question = question[0] if question else ""
        elif question is None:
            question = ""
        else:
            question = str(question)

        return question.strip()

    def _process_stage1_batch(batch_data: Tuple) -> Dict[str, Any]:
        """Process batch for stage 1 training."""
        docs_list, questions, answers, data_types, pos_indices = batch_data
        B = len(questions) 
        normalized_queries = [
            _normalize_stage1_query(question, data_type)
            for question, data_type in zip(questions, data_types)
        ]
       
       
        # Flatten documents for encoding
        # flat_docs = [doc for doc_list in docs_list for doc in doc_list] #原先的实现
        # print(len(flat_docs))
        #################################
        flat_docs = [] #change
        flat_questions = []
        for doc_list, query in zip(docs_list, normalized_queries):
            for doc in doc_list:
                flat_docs.append(doc)
                flat_questions.append(query)
        #################################
        # Prepare encoder inputs
        enc_inputs = clara_model._prepare_encoder_inputs(flat_docs, max_length=enc_max_len, q_texts=flat_questions) # 传入query
        enc_input_ids = enc_inputs["input_ids"]
        enc_attention_mask = enc_inputs["attention_mask"]
        
        assert enc_input_ids.size(0) == B * generation_top_k
        
        # Prepare decoder inputs
        prompt_responses = []
        for q, a, data_type in zip(normalized_queries, answers, data_types):
            if data_type == "paraphrase":
                # Handle paraphrase data (answer is a list)
                prompt_responses.append(
                    clara_model._blend_prompt_and_memory_tokens(
                        query=q, answer=a[0], paraphrase_loss=True, stage=clara_model.training_stage
                    )
                )
            else:
                prompt_responses.append(
                    clara_model._blend_prompt_and_memory_tokens(
                        query=q, answer=a, qa_loss=qa_loss, stage=clara_model.training_stage
                    )
                )
        
        prompt_lengths = [pr[0] for pr in prompt_responses]
        instructions = [pr[1] for pr in prompt_responses]
        
        # Tokenize decoder inputs
        dec_inputs = tokenizer(
            instructions,
            return_tensors="pt",
            padding="longest",
            add_special_tokens=False,
            truncation=True,
            max_length=dec_max_len,
        )
        
        dec_input_ids = dec_inputs["input_ids"]
        dec_attention_mask = dec_inputs["attention_mask"]
        dec_input_ids, dec_attention_mask = _pad_to_block_multiple(dec_input_ids, dec_attention_mask, tokenizer.pad_token_id)
        
        # Create labels
        labels = torch.where(
            dec_attention_mask.bool(),
            dec_input_ids.clone(),
            torch.tensor(-100, dtype=dec_input_ids.dtype),
        )
        labels = _mask_prompt(labels, dec_attention_mask, prompt_lengths, -100)
        
        return {
            "stage": clara_model.training_stage,
            "enc_input_ids": enc_input_ids,
            "enc_attention_mask": enc_attention_mask,
            "dec_input_ids": dec_input_ids,
            "dec_attention_mask": dec_attention_mask,
            "labels": labels,
            "questions": questions,
            "normalized_queries": normalized_queries,
            "answers": answers,
            "docs": docs_list,
        }
    
    def collate(batch: List[Tuple]) -> Dict[str, Any]:
        """Main collate function that routes to appropriate stage processor."""
        # Unpack batch
        docs_list, questions, answers, data_types, pos_indices = zip(*batch)
        
        # Convert to lists
        docs_list = list(docs_list)
        questions = list(questions)
        answers = list(answers)
        data_types = list(data_types)
        pos_indices = list(pos_indices)
        
        assert len(docs_list[0]) == generation_top_k, \
            f"Expected {generation_top_k} documents, got {len(docs_list[0])}"
        
        batch_data = (docs_list, questions, answers, data_types, pos_indices)
        
        # Route to appropriate processor
        if clara_model.training_stage in ["stage1", "stage1_2"]:
            return _process_stage1_batch(batch_data)
        else:
            raise ValueError(f"Unsupported training stage for this build: {clara_model.training_stage}")
    
    return collate


def preprocess_data(data: Dict[str, Any], 
                   input_template: Optional[str] = None,
                   input_key: str = "input",
                   output_key: Optional[str] = None,
                   apply_chat_template: Optional[Callable] = None,
                   multiturn: bool = False) -> Tuple[List[str], str, str, str, List[int]]:
    """
    Preprocess raw data into format expected by CLaRa dataset.
    
    Args:
        data: Raw data dictionary
        input_template: Template for input formatting
        input_key: Key for input data
        output_key: Key for output data
        apply_chat_template: Chat template function
        multiturn: Whether this is multiturn data
        
    Returns:
        Tuple of (docs, question, answer, data_type, pos_index)
    """
    # Extract documents. MRQA plain_text stores one context as a string,
    # while CLaRa expects a list of documents for generation_top_k.
    if "docs" in data and isinstance(data['docs'], list):
        docs = data['docs']
    elif "context" in data and isinstance(data['context'], list):
        docs = data['context']
    elif "context" in data and isinstance(data['context'], str):
        docs = [data['context']]
    elif "content" in data and isinstance(data['content'], list):
        docs = data['content']
    elif "content" in data and isinstance(data['content'], str):
        docs = [data['content']]
    else:
        raise ValueError(f"No valid document field found in data: {list(data.keys())}")
    
    # Extract answers
    if "answer" in data and isinstance(data['answer'], str):
        answers = data['answer']
    elif "answers" in data and isinstance(data['answers'], list) and isinstance(data.get('question'), list):
        answers = data['answers']
    elif "answers" in data and isinstance(data['answers'], list) and data['answers']:
        answers = data['answers'][0]
    elif "golden_answers" in data and isinstance(data['golden_answers'], list):
        answers = data['golden_answers'][0]
    else:
        raise ValueError(f"No valid answer field found in data: {list(data.keys())}")
    
    # Extract data type
    data_type = data.get('data_type', 'qa')
    
    # Extract question
    questions = data.get('question', [])
    
    # Extract positive indices
    pos_index = data.get('pos_index', [])
    
    return docs, questions, answers, data_type, pos_index


class SFTDataset(Dataset):
    """
    Dataset for CLaRa Supervised Fine-Tuning.
    
    This dataset handles data preprocessing and loading for different CLaRa training stages.
    """
    
    def __init__(self,
                 dataset,
                 tokenizer: Callable,
                 max_length: int,
                 strategy,
                 input_template: Optional[str] = None,
                 pretrain_mode: bool = False,
                 num_processors: int = 8,
                 multiturn: bool = False) -> None:
        """
        Initialize the SFT dataset.
        
        Args:
            dataset: HuggingFace dataset object
            tokenizer: Tokenizer function
            max_length: Maximum sequence length
            strategy: Training strategy object
            input_template: Template for input formatting
            pretrain_mode: Whether in pretraining mode
            num_processors: Number of processors for data processing
            multiturn: Whether to handle multiturn conversations
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiturn = multiturn
        
        # Chat template configuration
        self.input_template = input_template
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)
        
        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(self.strategy.args, "tokenizer_chat_template", None)
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
        
        # Process dataset in parallel
        processed_dataset = dataset.map(
            self._process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        
        # Store processed data
        self.docs = processed_dataset["docs"]
        self.questions = processed_dataset["questions"]
        self.answers = processed_dataset["answers"]
        self.data_type = processed_dataset["data_type"]
        self.pos_index = processed_dataset["pos_index"]
    
    def _process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single data sample."""
        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None
        
        if self.multiturn:
            assert (
                not self.output_key or not data[self.output_key]
            ), "For multiturn data, put the whole trajectory in input_key and don't set output_key"
            
            # Process multiturn conversation
            input_key = self.input_key
            apply_chat_template = self.apply_chat_template
            response_ranges = []
            
            for idx, message in enumerate(data[input_key]):
                if message["role"] == "assistant":
                    prompt = apply_chat_template(
                        data[input_key][:idx], 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    response = apply_chat_template(
                        data[input_key][:idx + 1], 
                        tokenize=False
                    )[len(prompt):]
                    
                    # Calculate token ranges
                    start_idx = (
                        self.tokenizer(
                            prompt,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                    )
                    
                    end_idx = (
                        start_idx
                        + self.tokenizer(
                            response,
                            max_length=self.max_length,
                            padding=False,
                            truncation=True,
                            return_tensors="pt",
                            add_special_tokens=False,
                        )["attention_mask"]
                        .int()
                        .sum()
                        .item()
                        - 1
                    )
                    
                    response_ranges.append((start_idx, end_idx))
        
        # Preprocess the data
        docs, questions, answers, data_type, pos_index = preprocess_data(data)
        
        return {
            "docs": docs,
            "questions": questions,
            "answers": answers,
            "data_type": data_type,
            "pos_index": pos_index,
        }
    
    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Tuple[List[str], str, str, str, List[int]]:
        """Get a single item from the dataset."""
        return (
            self.docs[idx],
            self.questions[idx],
            self.answers[idx],
            self.data_type[idx],
            self.pos_index[idx]
        )


class ContrastSFTDataset(Dataset):
    """
    Dataset for in-batch contrastive learning.

    Some datasets store one context with multiple question-answer pairs. Those
    samples are unfolded so items in each group share the same context. Plain
    MRQA rows already contain one question-answer pair per sample, so they are
    exposed directly and contrastive batches are formed by the sampler.
    """

    def __init__(self,
                 dataset,
                 tokenizer,
                 max_length: int,
                 strategy,
                 num_questions: int = 8,
                 num_processors: int = 8) -> None:
        super().__init__()
        self.num_questions = num_questions

        processed_dataset = dataset.map(
            self._process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )

        self.docs = processed_dataset["docs"]
        self.questions = processed_dataset["questions"]
        self.answers = processed_dataset["answers"]
        self.data_type = processed_dataset["data_type"]
        self.pos_index = processed_dataset["pos_index"]
        first_question = self.questions[0] if len(self.questions) > 0 else None
        self.has_grouped_questions = isinstance(first_question, (list, tuple))

    @staticmethod
    def _process_data(data):
        docs, questions, answers, data_type, pos_index = preprocess_data(data)
        return {
            "docs": docs,
            "questions": questions,
            "answers": answers,
            "data_type": data_type,
            "pos_index": pos_index,
        }

    def __len__(self) -> int:
        if self.has_grouped_questions:
            return len(self.questions) * self.num_questions
        return len(self.questions)

    def __getitem__(self, idx: int):
        if not self.has_grouped_questions:
            return (
                self.docs[idx],
                self.questions[idx],
                self.answers[idx],
                self.data_type[idx],
                self.pos_index[idx],
            )

        context_idx = idx // self.num_questions
        question_idx = idx % self.num_questions
        return (
            self.docs[context_idx],
            self.questions[context_idx][question_idx],
            self.answers[context_idx][question_idx],
            self.data_type[context_idx],
            self.pos_index[context_idx],
        )