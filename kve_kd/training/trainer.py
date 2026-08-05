import math
import torch
import torch.nn as nn
from transformers import Trainer
from typing import List, Optional, Tuple
from torch.utils.data import Sampler
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

try:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.trainer import ALL_LAYERNORM_LAYERS

try:
    from transformers.trainer_pt_utils import get_parameter_names
except ImportError:
    from transformers.trainer import get_parameter_names

from transformers.trainer import has_length, logger
from transformers.utils import is_sagemaker_mp_enabled, is_apex_available

try:
    from transformers.trainer_utils import ShardedDDPOption
except ImportError:
    from enum import Enum
    class ShardedDDPOption(Enum):
        SIMPLE = "simple"
        ZERO_DP_2 = "zero_dp_2"
        ZERO_DP_3 = "zero_dp_3"
        OFFLOAD = "offload"
import torch.nn.functional as F

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
if is_apex_available():
    from apex import amp

try:
    from fairscale.optim import OSS
except ImportError:
    OSS = None

import os


def split_to_even_chunks(indices, lengths, num_chunks):
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)] 

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    assert all(l != 0 for l in lengths), "Should not have zero length."
    
    mm_pairs = [(i, l) for i, l in enumerate(lengths) if l > 0]
    lang_pairs = [(i, -l) for i, l in enumerate(lengths) if l < 0]
    
    if len(mm_pairs) == 0 and len(lang_pairs) == 0:
        raise ValueError("Dataset has no valid samples (all lengths are zero).")
    
    if len(lang_pairs) == 0:
        mm_indices, mm_lengths = zip(*mm_pairs)
        return get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=generator)
    
    if len(mm_pairs) == 0:
        lang_indices, lang_lengths = zip(*lang_pairs)
        return get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=generator)
    
    mm_indices, mm_lengths = zip(*mm_pairs)
    lang_indices, lang_lengths = zip(*lang_pairs)

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=generator)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=generator)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches] 
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
        drop_last: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality
        self.drop_last = drop_last
        
        total_samples = len(self.lengths)
        megabatch_size = self.world_size * self.batch_size
        if self.drop_last:
            self._effective_length = (total_samples // megabatch_size) * megabatch_size
        else:
            self._effective_length = total_samples

    def __len__(self):
        return self._effective_length

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        
        if self.drop_last:
            indices = indices[:self._effective_length]
        
        return iter(indices)

    
class VLMTrainer(Trainer):
    
    def __init__(self,
        model = None,
        teacher = None,
        args = None,
        data_collator = None,
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        model_init = None,
        compute_metrics = None,
        callbacks = None,
        optimizers = (None, None),
        preprocess_logits_for_metrics = None,
        ):
        super(VLMTrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        if args.distill==1:
            self.teacher = teacher
            
            model_type = getattr(args, 'model_type', 'mobilellama')
            if model_type == 'qwen3_vl':
                teacher_heads = 32
                student_heads = 16
                teacher_dim = 4096
                student_dim = 2048
            else:
                teacher_heads = 32
                student_heads = 16
                teacher_dim = 4096
                student_dim = 2048
            
            # Use student/training device for adapters to avoid cross-device issues
            train_device = next(model.parameters()).device
            train_dtype = next(model.parameters()).dtype
            
            self.attn_adapter = torch.nn.Sequential(
                torch.nn.Conv2d(teacher_heads, student_heads, 1),
            ).to(device=train_device, dtype=train_dtype).train()
            self.proj_adapter = torch.nn.Sequential(
                torch.nn.Conv1d(teacher_dim, student_dim, 1),
            ).to(device=train_device, dtype=train_dtype).train()
            
            self._original_proj_adapter = self.proj_adapter
            self._original_attn_adapter = self.attn_adapter
            
            self.model.attn_adapter = self.attn_adapter
            self.model.proj_adapter = self.proj_adapter
        
    def _get_train_sampler(self, dataset=None) -> Optional[torch.utils.data.Sampler]:
        train_dataset = dataset if dataset is not None else self.train_dataset
        
        if train_dataset is None or not has_length(train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
                drop_last=self.args.dataloader_drop_last,
            )
        else:
            import inspect
            sig = inspect.signature(super()._get_train_sampler)
            if 'dataset' in sig.parameters:
                return super()._get_train_sampler(dataset=train_dataset)
            else:
                return super()._get_train_sampler()

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        sharded_ddp = getattr(self, 'sharded_ddp', None)
        if sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(
                opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [
                name for name in decay_parameters if "bias" not in name]

            unused_parameters = [
                name for name, _ in opt_model.named_parameters() if "vision_tower" in name and "layers" not in name
            ]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters()
                    if "mm_projector" in name
                    or "visual.merger" in name
                    or "visual.deepstack_merger_list" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)
            
            if sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs)

                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel()
                                           for p in module.parameters()}.values())
                            logger.info(
                                f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(
                                module, "weight", {"optim_bits": 32})
                            logger.debug(
                                f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None):
        torch.cuda.empty_cache()
        model.train()
        
        idx = inputs.pop('idx')
        inputs = self._prepare_inputs(inputs)
        inputs['idx']=idx
        
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        torch.cuda.empty_cache()
        
        # Ensure loss is scalar before backward (required by DeepSpeed)
        if isinstance(loss, torch.Tensor):
            if loss.numel() != 1 or loss.dim() != 0:
                loss = loss.mean()
            if loss.dim() != 0:
                loss = loss.view([])
        
        if getattr(self, 'do_grad_scaling', False) or (hasattr(self, 'scaler') and self.scaler is not None):
            self.scaler.scale(loss).backward()
        elif getattr(self, 'use_apex', False):
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        
        
        torch.cuda.empty_cache()

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def log(self, logs, start_time=None, **kwargs):
        if (self.args.distill == 1 and 
            hasattr(self, '_loss_count') and 
            self._loss_count > 0 and
            'loss' in logs):
            
            task = round(self._loss_task_sum / self._loss_count, 4)
            rkld = round(self._loss_rkld_sum / self._loss_count, 4)
            vision = round(self._loss_vision_sum / self._loss_count, 4)
            attn = round(self._loss_attn_sum / self._loss_count, 4)
            
            logs['sub'] = f"task: {task}, kl: {rkld}, Vis: {vision}, attn: {attn}"
            
            self._loss_task_sum = 0.0
            self._loss_rkld_sum = 0.0
            self._loss_vision_sum = 0.0
            self._loss_attn_sum = 0.0
            self._loss_count = 0
        
        if (getattr(self.args, 'key_vis_token_method', 'attn_topk') == 'attn_topk' and
            hasattr(self, '_attn_topk_count') and self._attn_topk_count > 0):
            avg_selected = self._attn_topk_selected_sum / self._attn_topk_count
            logs['attn_topk'] = f"sel={avg_selected:.1f}"
            
            self._attn_topk_selected_sum = 0.0
            self._attn_topk_count = 0
        
        if (getattr(self.args, 'key_vis_token_method', 'attn_topk') == 'kve_topk' and
            hasattr(self, '_kve_topk_count') and self._kve_topk_count > 0):
            avg_trigger = self._kve_topk_trigger_sum / self._kve_topk_count
            avg_selected = self._kve_topk_selected_sum / self._kve_topk_count
            avg_layer = (self._kve_topk_layer_sum / self._kve_topk_layer_count
                         if self._kve_topk_layer_count > 0 else -1)
            logs['kve_topk'] = f"trig={avg_trigger:.2f}, sel={avg_selected:.1f}, layer={avg_layer:.1f}"

            self._kve_topk_trigger_sum = 0.0
            self._kve_topk_selected_sum = 0.0
            self._kve_topk_layer_sum = 0.0
            self._kve_topk_layer_count = 0
            self._kve_topk_count = 0

        if (getattr(self.args, 'key_vis_token_method', 'attn_topk') == 'cosine_l2' and
            hasattr(self, '_cosine_l2_count') and self._cosine_l2_count > 0):
            avg_trigger = self._cosine_l2_trigger_sum / self._cosine_l2_count
            avg_selected = self._cosine_l2_selected_sum / self._cosine_l2_count
            avg_layer = (self._cosine_l2_layer_sum / self._cosine_l2_layer_count 
                         if self._cosine_l2_layer_count > 0 else -1)
            logs['cosine_l2'] = f"trig={avg_trigger:.2f}, sel={avg_selected:.1f}, layer={avg_layer:.1f}"
            
            self._cosine_l2_trigger_sum = 0.0
            self._cosine_l2_selected_sum = 0.0
            self._cosine_l2_layer_sum = 0.0
            self._cosine_l2_layer_count = 0
            self._cosine_l2_count = 0
        
        logs.pop('grad_norm', None)
        
        if 'learning_rate' in logs:
            logs['learning_rate'] = float(f"{logs['learning_rate']:.6g}")
        
        super().log(logs, start_time=start_time, **kwargs)
    
    def get_distil_loss(self, logits, teacher_logits):
        compute_dtype = logits.dtype
        
        student_probs = F.softmax(logits, dim=-1, dtype=compute_dtype)
        
        inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
        
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=compute_dtype)
        
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=compute_dtype)
        
        prod_probs = torch.masked_fill(
            student_probs * (student_logprobs - teacher_logprobs),
            inf_mask,
            0
        )
        
        x = torch.sum(prod_probs, dim=-1).view(-1)
        
        distil_loss = torch.mean(x, dim=0)
        
        return distil_loss

    def compute_adaptive_k(self, teacher_front_attn, t_v_mask):
        batch_size = teacher_front_attn.shape[0]
        
        mask = (t_v_mask[:, 0] != 0)
        
        A = teacher_front_attn.mean(1)
        
        scores = (A * mask).sum(dim=-2)
        
        vision_cols = mask.any(dim=-2)
        
        k_values = []
        H_normalized_list = []
        
        for b in range(batch_size):
            vis_scores = scores[b, vision_cols[b]]
            
            if vis_scores.numel() == 0:
                k_values.append(self.args.topk_min)
                H_normalized_list.append(torch.tensor(0.5, device=scores.device))
                continue
                
            p = vis_scores / (vis_scores.sum() + 1e-10)
            
            p_log_p = p * torch.log(p.clamp(min=1e-10))
            H = -p_log_p.sum()
            
            N_vis = vis_scores.numel()
            H_normalized = H / (torch.log(torch.tensor(float(N_vis), dtype=H.dtype, device=H.device)) + 1e-10)
            H_normalized = H_normalized.clamp(0, 1)
            H_normalized_list.append(H_normalized)
            
            k_range = self.args.topk_max - self.args.topk_min
            k = self.args.topk_min + k_range * (H_normalized ** self.args.topk_gamma)
            k = int(k.item())
            
            k = max(self.args.topk_min, min(k, self.args.topk_max, N_vis))
            k_values.append(k)
        
        k_tensor = torch.tensor(k_values, dtype=torch.long, device=teacher_front_attn.device)
        H_tensor = torch.stack(H_normalized_list)
        
        return k_tensor, H_tensor

    def get_vision_positions(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor):
        """
        Dynamically identify vision token positions for each sample.
        
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
        
        Returns:
            vision_positions_list: List[Tensor], vision token position indices for each sample
            q_idx_list: List[int], last valid text token position for each sample
        """
        from kve_kd.constants import (
            QWEN3_VISION_START_ID, QWEN3_VISION_END_ID,
            QWEN3_IMAGE_TOKEN_ID, QWEN3_VIDEO_TOKEN_ID,
        )
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        vision_positions_list = []
        q_idx_list = []
        
        special_ids = {QWEN3_VISION_START_ID, QWEN3_VISION_END_ID, 
                       QWEN3_IMAGE_TOKEN_ID, QWEN3_VIDEO_TOKEN_ID}
        
        for b in range(batch_size):
            ids = input_ids[b]
            
            # Identify vision token range (aligned with _extract_vision_features)
            starts = (ids == QWEN3_VISION_START_ID).nonzero(as_tuple=True)[0]
            ends = (ids == QWEN3_VISION_END_ID).nonzero(as_tuple=True)[0]
            
            vision_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            for i in range(min(len(starts), len(ends))):
                st = int(starts[i].item())
                ed = int(ends[i].item())
                if st < ed:
                    vision_mask[st + 1 : ed] = True
            
            # fallback: aligned with _extract_vision_features
            if not vision_mask.any():
                vision_mask = (ids == QWEN3_IMAGE_TOKEN_ID) | (ids == QWEN3_VIDEO_TOKEN_ID)
            
            if attention_mask is not None:
                vision_mask = vision_mask & attention_mask[b].bool()
            
            vision_positions = vision_mask.nonzero(as_tuple=True)[0]
            vision_positions_list.append(vision_positions)
            
            # Calculate last valid text token position (excluding vision and special tokens)
            valid_pos = (attention_mask[b] == 1).nonzero(as_tuple=True)[0]
            
            special_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            for sid in special_ids:
                special_mask = special_mask | (ids == sid)
            
            # Text tokens = valid positions - vision tokens - special tokens
            text_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
            text_mask[vision_positions] = False
            text_mask = text_mask & (~special_mask)
            text_pos = valid_pos[text_mask[valid_pos]]
            
            if len(text_pos) > 0:
                q_idx = text_pos.max().item()
            elif len(valid_pos) > 0:
                q_idx = valid_pos.max().item()  # fallback: last valid position
            else:
                q_idx = seq_len - 1  # fallback
            
            q_idx_list.append(q_idx)
        
        return vision_positions_list, q_idx_list

    def select_key_vis_tokens_cosine_l2(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        teacher_attentions: Tuple[torch.Tensor],
        teacher_past_key_values,
        key_vis_p: float,
        key_vis_layer_threshold: float,
        key_vis_tokens_threshold: float,
    ):
        """
        Layer-wise visual token selection using cosine similarity and L2 thresholds.
        
        Args:
            input_ids: [B, S]
            attention_mask: [B, S]
            teacher_attentions: tuple of [B, H, S, S], attention weights for all layers
            teacher_past_key_values: DynamicCache or tuple, key/value states for all layers
            key_vis_p: quantile ratio for filtering layer trigger
            key_vis_layer_threshold: cosine threshold for filtering layer
            key_vis_tokens_threshold: L2 threshold for key token selection
        
        Returns:
            selected_indices: List[Tensor], selected vision token indices for each sample
            filtering_layer_indices: List[int], filtering layer index for each sample (-1 if not triggered)
            stats: dict, statistics
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        num_layers = len(teacher_attentions)
        
        vision_positions_list, q_idx_list = self.get_vision_positions(input_ids, attention_mask)
        
        selected_indices = []
        filtering_layer_indices = []
        stats_per_sample = []
        
        for b in range(batch_size):
            vision_pos = vision_positions_list[b]
            n_vis = len(vision_pos)
            q_idx = q_idx_list[b]
            
            if n_vis == 0:
                selected_indices.append(torch.arange(0, device=device))
                filtering_layer_indices.append(-1)
                stats_per_sample.append({'triggered': False, 'layer': -1, 'n_selected': 0, 'n_vis': 0})
                continue
            
            triggered = False
            triggered_layer = -1
            sample_selected = None
            
            for layer_idx in range(num_layers):
                attn_weights = teacher_attentions[layer_idx]  # [B, H, S, S]
                
                # Support multiple cache formats: DynamicCache, Cache, legacy tuple
                value_states = None
                cache_type_name = type(teacher_past_key_values).__name__
                
                if 'Cache' in cache_type_name:
                    # DynamicCache / Cache format (transformers >= 4.36)
                    if hasattr(teacher_past_key_values, 'value_cache'):
                        value_states = teacher_past_key_values.value_cache[layer_idx]
                    elif hasattr(teacher_past_key_values, 'get_seq_length'):
                        kv_pair = teacher_past_key_values[layer_idx]
                        if isinstance(kv_pair, tuple) and len(kv_pair) >= 2:
                            value_states = kv_pair[1]
                
                if value_states is None:
                    if isinstance(teacher_past_key_values, (list, tuple)):
                        # Legacy tuple format: ((k, v), (k, v), ...)
                        kv_pair = teacher_past_key_values[layer_idx]
                        if isinstance(kv_pair, tuple) and len(kv_pair) >= 2:
                            value_states = kv_pair[1]
                        else:
                            value_states = kv_pair
                
                if value_states is None:
                    raise ValueError(f"Cannot extract value_states from past_key_values type: {type(teacher_past_key_values)}")
                
                num_heads = attn_weights.shape[1]
                num_kv_heads = value_states.shape[1]
                
                # GQA: repeat value states to match attention heads
                if num_kv_heads != num_heads:
                    n_rep = num_heads // num_kv_heads
                    value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)
                else:
                    value_states_expanded = value_states
                
                attn_w_q = attn_weights[b, :, q_idx, :]  # [H, S]
                value_b = value_states_expanded[b]  # [H, S, D]
                
                O_q = (attn_w_q.unsqueeze(-1) * value_b).sum(dim=1)  # [H, D]
                O_q_flat = O_q.view(-1)  # [H*D]
                
                attn_w_vis = attn_w_q[:, vision_pos]  # [H, N_vis]
                value_vis = value_b[:, vision_pos, :]  # [H, N_vis, D]
                contrib_vis = attn_w_vis.unsqueeze(-1) * value_vis  # [H, N_vis, D]
                contrib_vis = contrib_vis.permute(1, 0, 2).contiguous()  # [N_vis, H, D]
                contrib_vis_flat = contrib_vis.view(n_vis, -1)  # [N_vis, H*D]
                
                O_masked = O_q_flat.unsqueeze(0) - contrib_vis_flat  # [N_vis, H*D]
                cos_sim = F.cosine_similarity(O_masked, O_q_flat.unsqueeze(0), dim=-1)  # [N_vis]
                
                # Filtering layer judgment (quantile + minimum count lower bound)
                k = max(1, int(math.ceil(key_vis_p * n_vis)))
                bottom_k_cos, _ = cos_sim.topk(k, largest=False)
                q_val = bottom_k_cos.max()
                
                if q_val < key_vis_layer_threshold:
                    l2 = torch.norm(O_masked - O_q_flat.unsqueeze(0), p=2, dim=-1)  # [N_vis]
                    important_mask = l2 > key_vis_tokens_threshold
                    important_indices = important_mask.nonzero(as_tuple=True)[0]
                    
                    if len(important_indices) == 0:
                        important_indices = torch.arange(n_vis, device=device)
                    
                    sample_selected = important_indices
                    triggered = True
                    triggered_layer = layer_idx
                    break
            
            if not triggered:
                sample_selected = torch.arange(n_vis, device=device)
            
            selected_indices.append(sample_selected)
            filtering_layer_indices.append(triggered_layer)
            stats_per_sample.append({
                'triggered': triggered,
                'layer': triggered_layer,
                'n_selected': len(sample_selected),
                'n_vis': n_vis,
            })
        
        triggered_count = sum(1 for s in stats_per_sample if s['triggered'])
        triggered_layers = [s['layer'] for s in stats_per_sample if s['triggered']]
        
        stats = {
            'trigger_rate': triggered_count / batch_size if batch_size > 0 else 0,
            'avg_selected': sum(len(idx) for idx in selected_indices) / batch_size if batch_size > 0 else 0,
            'avg_trigger_layer': sum(triggered_layers) / len(triggered_layers) if triggered_layers else -1,
            'per_sample': stats_per_sample,
        }
        
        return selected_indices, filtering_layer_indices, stats

    def _get_kve_query_positions(
        self,
        q_positions: torch.Tensor,
        fallback_q_idx: int,
        key_vis_kve_query_mode: str,
    ) -> torch.Tensor:
        """Resolve the query positions used by kve_topk."""
        if key_vis_kve_query_mode == "last_text_token":
            if q_positions.numel() > 0:
                return q_positions[-1:].contiguous()
            return torch.tensor([fallback_q_idx], device=q_positions.device, dtype=torch.long)

        if key_vis_kve_query_mode == "all_text_sum":
            if q_positions.numel() > 0:
                return q_positions
            return torch.tensor([fallback_q_idx], device=q_positions.device, dtype=torch.long)

        raise ValueError(
            f"Unsupported key_vis_kve_query_mode: {key_vis_kve_query_mode}. "
            "Expected one of: last_text_token, all_text_sum"
        )

    def _aggregate_kve_query_attention(
        self,
        attn_weights_layer: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate query attention for kve_topk.
        attn_weights_layer: [H, S, S] for one sample and one layer.
        Returns aggregated attention [H, S].
        """
        if query_positions.numel() == 0:
            raise ValueError("query_positions must not be empty for kve_topk aggregation")
        return attn_weights_layer[:, query_positions, :].sum(dim=1)

    def _compute_adaptive_k_kve(self, vis_scores, topk_min, topk_max, topk_gamma):
        """
        Compute adaptive k based on entropy of the aggregated visual attention distribution.
        """
        n_vis = vis_scores.numel()
        if n_vis == 0:
            return topk_min
        score_sum = vis_scores.sum()
        if score_sum.item() <= 0:
            return topk_min
        p = vis_scores / (score_sum + 1e-10)
        H = -(p * torch.log(p.clamp(min=1e-10))).sum()
        H_norm = (H / (torch.log(torch.tensor(float(n_vis), dtype=H.dtype, device=H.device)) + 1e-10)).clamp(0, 1)
        k = int((topk_min + (topk_max - topk_min) * (H_norm ** topk_gamma)).item())
        return max(topk_min, min(k, topk_max, n_vis))

    def select_key_vis_tokens_kve_topk(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        teacher_attentions,
        teacher_past_key_values,
        key_vis_trigger_metric: str,
        key_vis_layer_threshold: float,
        key_vis_rel_l2_layer_threshold: float,
        key_vis_trigger_scan_start_layer: int,
        key_vis_kve_query_mode: str,
    ):
        """
        Fusion-Triggered Attention TopK selection.
        Stage 1: scan layers to find first trigger layer using the configured query mode.
        Stage 2: rank visual tokens by the same query-mode-aggregated attention in trigger layer, top-k.
        Returns selected_indices, trigger_layer_indices, stats (compatible with cosine_l2 interface).
        """
        from kve_kd.constants import (
            IGNORE_INDEX,
            QWEN3_VISION_START_ID, QWEN3_VISION_END_ID,
            QWEN3_IMAGE_TOKEN_ID, QWEN3_VIDEO_TOKEN_ID,
        )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        num_layers = len(teacher_attentions)
        special_ids = {QWEN3_VISION_START_ID, QWEN3_VISION_END_ID,
                       QWEN3_IMAGE_TOKEN_ID, QWEN3_VIDEO_TOKEN_ID}

        selected_indices = []
        trigger_layer_indices = []
        stats_per_sample = []

        for b in range(batch_size):
            ids = input_ids[b]

            starts = (ids == QWEN3_VISION_START_ID).nonzero(as_tuple=True)[0]
            ends = (ids == QWEN3_VISION_END_ID).nonzero(as_tuple=True)[0]
            vision_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            for i in range(min(len(starts), len(ends))):
                st, ed = int(starts[i].item()), int(ends[i].item())
                if st < ed:
                    vision_mask[st + 1:ed] = True
            if not vision_mask.any():
                vision_mask = (ids == QWEN3_IMAGE_TOKEN_ID) | (ids == QWEN3_VIDEO_TOKEN_ID)
            if attention_mask is not None:
                vision_mask = vision_mask & attention_mask[b].bool()
            vision_pos = vision_mask.nonzero(as_tuple=True)[0]
            n_vis = vision_pos.numel()

            if n_vis == 0:
                selected_indices.append(torch.arange(0, device=device))
                trigger_layer_indices.append(-1)
                stats_per_sample.append({'triggered': False, 'layer': -1, 'n_selected': 0, 'n_vis': 0})
                continue

            # q*: last prefill text token (labels == IGNORE_INDEX, non-vision, non-special)
            special_mask_b = torch.zeros(seq_len, dtype=torch.bool, device=device)
            for sid in special_ids:
                special_mask_b = special_mask_b | (ids == sid)
            prefill_mask = (labels[b] == IGNORE_INDEX)
            valid_mask = (attention_mask[b] == 1)
            non_vis_non_special = (~vision_mask) & (~special_mask_b)
            q_positions = (prefill_mask & valid_mask & non_vis_non_special).nonzero(as_tuple=True)[0]
            if len(q_positions) > 0:
                fallback_q_idx = q_positions.max().item()
            else:
                fallback = (valid_mask & non_vis_non_special).nonzero(as_tuple=True)[0]
                fallback_q_idx = fallback.max().item() if len(fallback) > 0 else int((valid_mask).nonzero(as_tuple=True)[0].max().item())
            query_positions = self._get_kve_query_positions(
                q_positions=q_positions,
                fallback_q_idx=fallback_q_idx,
                key_vis_kve_query_mode=key_vis_kve_query_mode,
            )

            triggered = False
            triggered_layer = -1
            triggered_attn_layer = None

            for layer_idx in range(num_layers):
                if layer_idx < key_vis_trigger_scan_start_layer:
                    continue

                attn_weights = teacher_attentions[layer_idx]  # [B, H, S, S]
                attn_weights_b = attn_weights[b]              # [H, S, S]

                # Extract value states (reuse same logic as select_key_vis_tokens_cosine_l2)
                value_states = None
                cache_type_name = type(teacher_past_key_values).__name__
                if 'Cache' in cache_type_name:
                    if hasattr(teacher_past_key_values, 'value_cache'):
                        value_states = teacher_past_key_values.value_cache[layer_idx]
                    elif hasattr(teacher_past_key_values, 'get_seq_length'):
                        kv_pair = teacher_past_key_values[layer_idx]
                        if isinstance(kv_pair, tuple) and len(kv_pair) >= 2:
                            value_states = kv_pair[1]
                if value_states is None:
                    if isinstance(teacher_past_key_values, (list, tuple)):
                        kv_pair = teacher_past_key_values[layer_idx]
                        if isinstance(kv_pair, tuple) and len(kv_pair) >= 2:
                            value_states = kv_pair[1]
                        else:
                            value_states = kv_pair
                if value_states is None:
                    raise ValueError(f"Cannot extract value_states from past_key_values type: {type(teacher_past_key_values)}")

                num_heads = attn_weights.shape[1]
                num_kv_heads = value_states.shape[1]
                if num_kv_heads != num_heads:
                    value_states_exp = value_states.repeat_interleave(num_heads // num_kv_heads, dim=1)
                else:
                    value_states_exp = value_states

                attn_w_q = self._aggregate_kve_query_attention(attn_weights_b, query_positions)  # [H, S]
                value_b = value_states_exp[b]               # [H, S, D]
                O_q_flat = (attn_w_q.unsqueeze(-1) * value_b).sum(dim=1).view(-1)  # [H*D]

                attn_w_vis = attn_w_q[:, vision_pos]        # [H, N_vis]
                value_vis = value_b[:, vision_pos, :]       # [H, N_vis, D]
                contrib_vis_flat = (attn_w_vis.unsqueeze(-1) * value_vis).permute(1, 0, 2).contiguous().view(n_vis, -1)
                O_masked = O_q_flat.unsqueeze(0) - contrib_vis_flat  # [N_vis, H*D]

                if key_vis_trigger_metric == 'cosine':
                    trigger_hit = (F.cosine_similarity(O_masked, O_q_flat.unsqueeze(0), dim=-1) < key_vis_layer_threshold).any().item()
                else:  # rel_l2
                    rel_l2 = torch.norm(O_masked - O_q_flat.unsqueeze(0), p=2, dim=-1) / (O_q_flat.norm() + 1e-8)
                    trigger_hit = (rel_l2 > key_vis_rel_l2_layer_threshold).any().item()

                if trigger_hit:
                    triggered = True
                    triggered_layer = layer_idx
                    triggered_attn_layer = attn_weights_b  # [H, S, S]
                    break

            if not triggered:
                selected_indices.append(torch.arange(n_vis, device=device))
                trigger_layer_indices.append(-1)
                stats_per_sample.append({'triggered': False, 'layer': -1, 'n_selected': n_vis, 'n_vis': n_vis})
                continue

            attn_w_q_trigger = self._aggregate_kve_query_attention(triggered_attn_layer, query_positions)  # [H, S]
            r_vis = attn_w_q_trigger[:, vision_pos].mean(0)  # [n_vis]

            adaptive_topk = getattr(self.args, 'adaptive_topk', False)
            if adaptive_topk:
                k = self._compute_adaptive_k_kve(
                    r_vis,
                    self.args.topk_min,
                    self.args.topk_max,
                    self.args.topk_gamma,
                )
            else:
                k = getattr(self.args, 'topk_fixed', 16)
            k_eff = max(1, min(k, n_vis))

            _, topk_local_idx = r_vis.topk(k_eff)
            selected_indices.append(topk_local_idx)
            trigger_layer_indices.append(triggered_layer)
            stats_per_sample.append({'triggered': True, 'layer': triggered_layer, 'n_selected': k_eff, 'n_vis': n_vis})

        triggered_count = sum(1 for s in stats_per_sample if s['triggered'])
        triggered_layers = [s['layer'] for s in stats_per_sample if s['triggered']]
        stats = {
            'trigger_rate': triggered_count / batch_size if batch_size > 0 else 0,
            'avg_selected': sum(s['n_selected'] for s in stats_per_sample) / batch_size if batch_size > 0 else 0,
            'avg_trigger_layer': sum(triggered_layers) / len(triggered_layers) if triggered_layers else -1,
            'per_sample': stats_per_sample,
        }
        return selected_indices, trigger_layer_indices, stats

    def _get_v_loss_kve_topk(
        self,
        v_feature: torch.Tensor,
        teacher_v_feature: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        teacher_attentions,
        teacher_past_key_values,
    ):
        """KVE-TopK vision feature distillation loss (compatible with _get_v_loss_cosine_l2)."""
        batch_size = v_feature.shape[0]

        selected_indices, _, stats = self.select_key_vis_tokens_kve_topk(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            teacher_attentions=teacher_attentions,
            teacher_past_key_values=teacher_past_key_values,
            key_vis_trigger_metric=getattr(self.args, 'key_vis_trigger_metric', 'cosine'),
            key_vis_layer_threshold=self.args.key_vis_layer_threshold,
            key_vis_rel_l2_layer_threshold=getattr(self.args, 'key_vis_rel_l2_layer_threshold', 0.05),
            key_vis_trigger_scan_start_layer=getattr(self.args, 'key_vis_trigger_scan_start_layer', 0),
            key_vis_kve_query_mode=getattr(self.args, 'key_vis_kve_query_mode', 'last_text_token'),
        )

        if getattr(self.args, 'key_vis_debug', False):
            print(f"[kve_topk:{getattr(self.args, 'key_vis_kve_query_mode', 'last_text_token')}] trigger_rate={stats['trigger_rate']:.2f}, "
                  f"avg_selected={stats['avg_selected']:.1f}, "
                  f"avg_trigger_layer={stats['avg_trigger_layer']:.1f}")

        if not hasattr(self, '_kve_topk_trigger_sum'):
            self._kve_topk_trigger_sum = 0.0
            self._kve_topk_selected_sum = 0.0
            self._kve_topk_layer_sum = 0.0
            self._kve_topk_layer_count = 0
            self._kve_topk_count = 0
        self._kve_topk_trigger_sum += stats['trigger_rate']
        self._kve_topk_selected_sum += stats['avg_selected']
        if stats['avg_trigger_layer'] >= 0:
            self._kve_topk_layer_sum += stats['avg_trigger_layer']
            self._kve_topk_layer_count += 1
        self._kve_topk_count += 1

        max_vis = v_feature.shape[1]
        loss_pick_list = []
        loss_all_list = []

        for b in range(batch_size):
            v_idx = selected_indices[b]
            n_vis_real = stats['per_sample'][b]['n_vis']

            if len(v_idx) == 0 and n_vis_real > 0:
                v_idx = torch.arange(n_vis_real, device=v_feature.device)

            upper_bound = min(n_vis_real, max_vis) - 1 if n_vis_real > 0 else 0
            v_idx = v_idx.clamp(max=upper_bound) if len(v_idx) > 0 else v_idx

            if n_vis_real == 0 or len(v_idx) == 0:
                continue

            v_pick = v_feature[b, v_idx]
            t_v_pick = teacher_v_feature[b, v_idx]
            loss_pick = ((self.proj_adapter(t_v_pick.unsqueeze(0).permute(0, 2, 1)) -
                          v_pick.unsqueeze(0).permute(0, 2, 1)) ** 2).mean()
            loss_pick_list.append(loss_pick)

            loss_all = ((self.proj_adapter(teacher_v_feature[b:b+1].permute(0, 2, 1)) -
                         v_feature[b:b+1].permute(0, 2, 1)) ** 2).mean()
            loss_all_list.append(loss_all)

        if len(loss_pick_list) == 0:
            zero_loss = torch.tensor(0.0, device=v_feature.device, requires_grad=True)
            return zero_loss, zero_loss, None

        return torch.stack(loss_pick_list).mean(), torch.stack(loss_all_list).mean(), None

    def _get_v_loss_cosine_l2(
        self, 
        v_feature: torch.Tensor,
        teacher_v_feature: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        teacher_attentions: Tuple[torch.Tensor],
        teacher_past_key_values,
    ):
        """
        Vision feature distillation with layer-wise cosine-L2 token selection.
        """
        batch_size = v_feature.shape[0]
        
        selected_indices, filtering_layer_indices, stats = self.select_key_vis_tokens_cosine_l2(
            input_ids=input_ids,
            attention_mask=attention_mask,
            teacher_attentions=teacher_attentions,
            teacher_past_key_values=teacher_past_key_values,
            key_vis_p=self.args.key_vis_p,
            key_vis_layer_threshold=self.args.key_vis_layer_threshold,
            key_vis_tokens_threshold=self.args.key_vis_tokens_threshold,
        )
        
        if getattr(self.args, 'key_vis_debug', False):
            print(f"[cosine_l2] trigger_rate={stats['trigger_rate']:.2f}, "
                  f"avg_selected={stats['avg_selected']:.1f}, "
                  f"avg_trigger_layer={stats['avg_trigger_layer']:.1f}")
        
        if not hasattr(self, '_cosine_l2_trigger_sum'):
            self._cosine_l2_trigger_sum = 0.0
            self._cosine_l2_selected_sum = 0.0
            self._cosine_l2_layer_sum = 0.0
            self._cosine_l2_layer_count = 0
            self._cosine_l2_count = 0
        self._cosine_l2_trigger_sum += stats['trigger_rate']
        self._cosine_l2_selected_sum += stats['avg_selected']
        if stats['avg_trigger_layer'] >= 0:
            self._cosine_l2_layer_sum += stats['avg_trigger_layer']
            self._cosine_l2_layer_count += 1
        self._cosine_l2_count += 1
        
        loss_pick_list = []
        loss_all_list = []
        
        # Note: v_feature.shape[1] is max_vis in batch, not the actual vision token count per sample
        max_vis = v_feature.shape[1]
        
        for b in range(batch_size):
            v_idx = selected_indices[b]
            n_vis_real = stats['per_sample'][b]['n_vis']
            
            if len(v_idx) == 0 and n_vis_real > 0:
                v_idx = torch.arange(n_vis_real, device=v_feature.device)
            
            upper_bound = min(n_vis_real, max_vis) - 1 if n_vis_real > 0 else 0
            v_idx = v_idx.clamp(max=upper_bound) if len(v_idx) > 0 else v_idx
            
            if n_vis_real == 0 or len(v_idx) == 0:
                continue
            
            v_pick = v_feature[b, v_idx]
            t_v_pick = teacher_v_feature[b, v_idx]
            loss_pick = ((self.proj_adapter(t_v_pick.unsqueeze(0).permute(0,2,1)) - 
                         v_pick.unsqueeze(0).permute(0,2,1)) ** 2).mean()
            loss_pick_list.append(loss_pick)
            
            loss_all = ((self.proj_adapter(teacher_v_feature[b:b+1].permute(0,2,1)) - 
                        v_feature[b:b+1].permute(0,2,1)) ** 2).mean()
            loss_all_list.append(loss_all)
        
        if len(loss_pick_list) == 0:
            zero_loss = torch.tensor(0.0, device=v_feature.device, requires_grad=True)
            return zero_loss, zero_loss, None
        
        final_loss_pick = torch.stack(loss_pick_list).mean()
        final_loss_all = torch.stack(loss_all_list).mean()
        
        return final_loss_pick, final_loss_all, None

    def get_v_loss(self, teacher_front_attn, t_v_mask, v_feature, teacher_v_feature, k=None,
                   input_ids=None, attention_mask=None, labels=None, teacher_attentions=None,
                   teacher_past_key_values=None, key_vis_token_method="attn_topk"):
        """
        Compute vision feature distillation loss.
        
        When key_vis_token_method="attn_topk", fully preserve existing implementation.
        When key_vis_token_method="cosine_l2", use layer-wise cosine-L2 selection.
        When key_vis_token_method="kve_topk", use the existing key-value evidence TopK logic.
        """

        if key_vis_token_method == "cosine_l2":
            return self._get_v_loss_cosine_l2(
                v_feature, teacher_v_feature,
                input_ids, attention_mask,
                teacher_attentions, teacher_past_key_values
            )

        if key_vis_token_method == "kve_topk":
            return self._get_v_loss_kve_topk(
                v_feature, teacher_v_feature,
                input_ids, attention_mask, labels,
                teacher_attentions, teacher_past_key_values
            )
        
        batch_size = v_feature.shape[0]
        
        mask = (t_v_mask[:, 0] != 0)
        
        A = teacher_front_attn.mean(1)
        
        scores = (A * mask).sum(dim=-2)
        
        vision_cols = mask.any(dim=-2)
        
        H_normalized = None
        if k is None:
            if self.args.adaptive_topk:
                k_values, H_normalized = self.compute_adaptive_k(teacher_front_attn, t_v_mask)
            else:
                k_values = torch.full((batch_size,), self.args.topk_fixed, 
                                    dtype=torch.long, device=v_feature.device)
        elif isinstance(k, int):
            k_values = torch.full((batch_size,), k, 
                                dtype=torch.long, device=v_feature.device) 
        else:
            k_values = k
        
        loss_pick_list = []
        loss_all_list = []
        
        for b in range(batch_size):
            k_b = k_values[b].item()
            
            scores_masked = scores[b].masked_fill(~vision_cols[b], float('-inf'))
            k_eff = min(k_b, v_feature.shape[1])
            
            if k_eff <= 0:
                continue
                
            top_scores, top_cols = scores_masked.topk(k_eff, dim=-1)
            
            rank = torch.cumsum(vision_cols[b].to(torch.int64), dim=-1) - 1
            v_idx = torch.gather(rank, -1, top_cols).clamp(min=0, max=v_feature.shape[1]-1)
            
            v_pick = v_feature[b, v_idx]
            t_v_pick = teacher_v_feature[b, v_idx]
            
            loss_pick = ((self.proj_adapter(t_v_pick.unsqueeze(0).permute(0,2,1)) - 
                         v_pick.unsqueeze(0).permute(0,2,1)) ** 2).mean()
            loss_pick_list.append(loss_pick)
            
            loss_all = ((self.proj_adapter(teacher_v_feature[b:b+1].permute(0,2,1)) - 
                        v_feature[b:b+1].permute(0,2,1)) ** 2).mean()
            loss_all_list.append(loss_all)
        
        if len(loss_pick_list) == 0:
            zero_loss = torch.tensor(0.0, device=v_feature.device, requires_grad=True)
            return zero_loss, zero_loss, H_normalized
        
        final_loss_pick = torch.stack(loss_pick_list).mean()
        final_loss_all = torch.stack(loss_all_list).mean()
        
        avg_k_selected = sum(k_values).item() / batch_size if batch_size > 0 else 0
        if not hasattr(self, '_attn_topk_selected_sum'):
            self._attn_topk_selected_sum = 0.0
            self._attn_topk_count = 0
        self._attn_topk_selected_sum += avg_k_selected
        self._attn_topk_count += 1
        
        return final_loss_pick, final_loss_all, H_normalized
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Preserve labels for KVE semantic-anchor selection before they are popped.
        labels_for_kve = inputs.get('labels')
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        idx = inputs.pop('idx')
        if len(idx)==0:
            idx = None
        else:
            idx = idx[0]
        
        import os, time
        _profile = os.getenv("PROFILE_FORWARD", "0") == "1"
        if _profile:
            torch.cuda.synchronize()
            _t0 = time.perf_counter()
        
        model_out = model(**inputs)
        outputs, v_feature, topk_attn, first_attn, t_v_mask, _ = model_out
        
        if _profile:
            torch.cuda.synchronize()
            _t1 = time.perf_counter()

        teacher_past_key_values = None
        if self.args.distill == 1:
            with torch.no_grad():
                # Check if use_cache is needed (cosine_l2 and kve_topk both need value states)
                need_cache = getattr(self.args, 'key_vis_token_method', 'attn_topk') in ('cosine_l2', 'kve_topk')
                
                # Return value remains 6-tuple, unpacking unchanged
                teacher_out = self.teacher(**inputs, use_cache=need_cache)
                teacher_outputs, teacher_v_feature, teacher_topk_attn, teacher_first_attn, _, _ = teacher_out
                
                # Read cached value states for cosine_l2 and kve_topk.
                teacher_past_key_values = teacher_outputs.past_key_values if need_cache else None
                
                teacher_logits = teacher_outputs['logits']
            
            if _profile:
                torch.cuda.synchronize()
                _t2 = time.perf_counter()
                seq_len = inputs.get('input_ids', torch.tensor([[]])).shape[-1]
                print(f"[FORWARD TIMING] student={_t1-_t0:.3f}s teacher={_t2-_t1:.3f}s total={_t2-_t0:.3f}s seq_len={seq_len}")

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            # Get loss from model outputs (supports both dict and ModelOutput)
            if isinstance(outputs, dict):
                if "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                loss = outputs["loss"]
            else:
                # ModelOutput: access loss attribute directly instead of __getitem__(0)
                # which may return logits if loss is None
                loss = getattr(outputs, 'loss', None)
                if loss is None:
                    raise ValueError(
                        f"Model output does not contain loss. "
                        f"'labels' in inputs: {'labels' in inputs}, "
                        f"labels shape: {inputs.get('labels', 'N/A') if isinstance(inputs.get('labels'), torch.Tensor) else 'N/A'}. "
                        f"Ensure labels are provided and model is correctly configured."
                    )
            
            # Ensure loss is scalar (reduce if needed for both dict and ModelOutput)
            if loss is not None and loss.dim() > 0:
                loss = loss.mean()
            
            skip_distill = False
            if t_v_mask is None:
                skip_distill = True
            elif (t_v_mask != 0).sum() == 0:
                skip_distill = True
            if idx is None:
                skip_distill = True
                loss = loss * 0.0001
            
            if self.args.distill == 1:
                if skip_distill:
                    v_logits_distill = torch.tensor(0.0, device=loss.device)
                    v_loss = torch.tensor(0.0, device=loss.device)
                    attn_loss = torch.tensor(0.0, device=loss.device)
                    H_normalized = None
                else:
                    lambda_rkld = getattr(self.args, 'lambda_rkld', 1.0)
                    lambda_v_all = getattr(self.args, 'lambda_v_all', 1.0)
                    lambda_v_focus = getattr(self.args, 'lambda_v_focus', 0.1)
                    lambda_attn = getattr(self.args, 'lambda_attn', 1.0)
                    
                    v_logits_distill = lambda_rkld * self.get_distil_loss(outputs['logits'], teacher_logits)
                    
                    H_normalized = None
                    has_vision_tokens = (t_v_mask != 0).sum() > 0
                    if not has_vision_tokens:
                        v_loss = lambda_v_all * ((self.proj_adapter(teacher_v_feature.permute(0,2,1))-v_feature.permute(0,2,1))**2).mean()
                    else:
                        # Pass selection inputs required by cosine_l2 and kve_topk.
                        key_vis_method = getattr(self.args, 'key_vis_token_method', 'attn_topk')
                        extra_kwargs = {}
                        if key_vis_method == 'cosine_l2':
                            extra_kwargs = {
                                'input_ids': inputs.get('input_ids'),
                                'attention_mask': inputs.get('attention_mask'),
                                'teacher_attentions': teacher_outputs.attentions,
                                'teacher_past_key_values': teacher_past_key_values,
                                'key_vis_token_method': key_vis_method,
                            }
                        elif key_vis_method == 'kve_topk':
                            extra_kwargs = {
                                'input_ids': inputs.get('input_ids'),
                                'attention_mask': inputs.get('attention_mask'),
                                'labels': labels_for_kve,
                                'teacher_attentions': teacher_outputs.attentions,
                                'teacher_past_key_values': teacher_past_key_values,
                                'key_vis_token_method': key_vis_method,
                            }
                        
                        loss_v_focus, loss_v_all, H_normalized = self.get_v_loss(
                            teacher_topk_attn, t_v_mask, v_feature, teacher_v_feature, k=None,
                            **extra_kwargs
                        )
                        v_loss = lambda_v_all * loss_v_all + lambda_v_focus * loss_v_focus
                    
                    attn_loss = lambda_attn * ((self.attn_adapter(teacher_first_attn) - first_attn)[t_v_mask!=0]**2).mean()
                
                if not skip_distill and self.args.use_entropy_weighting and H_normalized is not None:
                    entropy_weight = (1.0 - H_normalized).mean().clamp(0.1, 1.0)
                    
                    v_logits_distill = entropy_weight * v_logits_distill
                    v_loss = entropy_weight * v_loss
                    attn_loss = entropy_weight * attn_loss

                total_loss = loss + v_logits_distill + v_loss + attn_loss
                
                if not hasattr(self, '_loss_task_sum'):
                    self._loss_task_sum = 0.0
                    self._loss_rkld_sum = 0.0
                    self._loss_vision_sum = 0.0
                    self._loss_attn_sum = 0.0
                    self._loss_count = 0
                
                self._loss_task_sum += loss.item()
                self._loss_rkld_sum += v_logits_distill.item()
                self._loss_vision_sum += v_loss.item()
                self._loss_attn_sum += attn_loss.item()
                self._loss_count += 1
                
                loss = total_loss
            torch.cuda.empty_cache()

        # Final scalar check before return (covers all code paths)
        if isinstance(loss, torch.Tensor) and (loss.numel() != 1 or loss.dim() != 0):
            loss = loss.mean()
            if loss.dim() != 0:
                loss = loss.view([])
        
        return (loss, outputs) if return_outputs else loss
