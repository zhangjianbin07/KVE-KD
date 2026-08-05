import os
import pathlib
import torch
import transformers
import warnings
from dataclasses import dataclass, field
from typing import Optional
import random
import numpy as np

# PyTorch < 2.6 can resume only from explicitly trusted local checkpoints.
if os.environ.get("KVE_KD_ALLOW_UNSAFE_TORCH_LOAD", "").lower() in {"1", "true", "yes"}:
    try:
        from transformers.utils import import_utils
        import transformers.trainer as trainer_mod
        if hasattr(import_utils, "check_torch_load_is_safe"):
            import_utils.check_torch_load_is_safe = lambda: None
        if hasattr(trainer_mod, "check_torch_load_is_safe"):
            trainer_mod.check_torch_load_is_safe = lambda: None
        warnings.warn(
            "KVE_KD_ALLOW_UNSAFE_TORCH_LOAD is enabled. Load checkpoints only from trusted sources.",
            RuntimeWarning,
        )
    except Exception as exc:
        warnings.warn(f"Failed to enable trusted-checkpoint compatibility: {exc}", RuntimeWarning)

from kve_kd.models.qwen3_vl import load_qwen3_vl_model
from kve_kd.data.qwen3_vl import make_qwen3_vl_data_module
from kve_kd.training.trainer import VLMTrainer


local_rank = None


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Student model path"}
    )
    teacher_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model path"}
    )


@dataclass
class DataArguments:
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Training data JSON file path"}
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "Image folder path"}
    )
    min_pixels: int = field(
        default=256 * 28 * 28,
        metadata={"help": "Minimum pixels"}
    )
    max_pixels: int = field(
        default=1280 * 28 * 28,
        metadata={"help": "Maximum pixels"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_max_length: int = field(default=2048)
    distill: int = field(default=1, metadata={"help": "Enable distillation (0=no, 1=yes)"})
    model_type: str = field(default="qwen3_vl", metadata={"help": "Model type"})
    task: str = field(default="pretrain", metadata={"help": "Task type (pretrain/finetune)"})
    
    lambda_rkld: float = field(default=1.0, metadata={"help": "Logits distillation weight"})
    lambda_v_all: float = field(default=1.0, metadata={"help": "Global vision loss weight"})
    lambda_v_focus: float = field(default=0.1, metadata={"help": "Focus vision loss weight"})
    lambda_attn: float = field(default=1.0, metadata={"help": "Attention distillation weight"})
    
    adaptive_topk: bool = field(default=False, metadata={"help": "Use adaptive K value"})
    topk_fixed: int = field(default=16, metadata={"help": "Fixed K value"})
    topk_min: int = field(default=8, metadata={"help": "K value lower bound"})
    topk_max: int = field(default=32, metadata={"help": "K value upper bound"})
    topk_gamma: float = field(default=1.0, metadata={"help": "Entropy mapping gamma parameter"})
    use_entropy_weighting: bool = field(default=False, metadata={"help": "Use entropy weighting"})
    topk_attn_layer_idx: int = field(
        default=0, 
        metadata={"help": "Attention layer index for topK selection, 0=first layer, -1=last layer"}
    )
    
    remove_unused_columns: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    mm_projector_lr: Optional[float] = field(default=0.0, metadata={"help": "mm_projector learning rate"})
    
    enable_mixed_attn: bool = field(
        default=False, 
        metadata={"help": "Enable mixed attention mode (layer 0 eager, others flash/sdpa)"}
    )
    
    # Key visual token selection parameters
    key_vis_token_method: str = field(
        default="attn_topk",
        metadata={"help": "Method for key visual token selection: attn_topk (default), cosine_l2, kve_topk"}
    )
    key_vis_p: float = field(
        default=0.01,
        metadata={"help": "Quantile ratio for filtering layer trigger (only for cosine_l2)"}
    )
    key_vis_layer_threshold: float = field(
        default=0.995,
        metadata={"help": "Cosine similarity threshold for filtering layer (only for cosine_l2)"}
    )
    key_vis_tokens_threshold: float = field(
        default=0.2,
        metadata={"help": "L2 norm threshold for key token selection (only for cosine_l2)"}
    )
    key_vis_debug: bool = field(
        default=False,
        metadata={"help": "Enable per-sample debug logging (only for cosine_l2 / kve_topk)"}
    )
    # kve_topk parameters
    key_vis_trigger_metric: str = field(
        default="cosine",
        metadata={"help": "Trigger metric for kve_topk: cosine (default) or rel_l2"}
    )
    key_vis_rel_l2_layer_threshold: float = field(
        default=0.05,
        metadata={"help": "Relative L2 threshold for kve_topk trigger (only when key_vis_trigger_metric=rel_l2)"}
    )
    key_vis_trigger_scan_start_layer: int = field(
        default=0,
        metadata={"help": "First layer index (0-based inclusive) to scan for kve_topk trigger"}
    )
    key_vis_kve_query_mode: str = field(
        default="last_text_token",
        metadata={"help": "Query mode for kve_topk: last_text_token (default) or all_text_sum"}
    )
    
    # LoRA parameters
    lora_enable: bool = field(default=False, metadata={"help": "Enable LoRA finetune"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_bias: str = field(default="none", metadata={"help": "LoRA bias: none/all/lora_only"})
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Target modules for LoRA, comma-separated"}
    )
    
    random_seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    missing_args = []
    if not model_args.model_name_or_path:
        missing_args.append("--model_name_or_path")
    if not data_args.data_path:
        missing_args.append("--data_path")
    if not data_args.image_folder:
        missing_args.append("--image_folder")
    if training_args.distill == 1 and not model_args.teacher_model_path:
        missing_args.append("--teacher_model_path")
    if missing_args:
        raise ValueError(f"Missing required argument(s): {', '.join(missing_args)}")
    
    random_seed = getattr(training_args, 'random_seed', 42)
    seed(random_seed)
    training_args.seed = random_seed  # sync HF Trainer's internal seed to avoid set_seed() override
    
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    
    enable_mixed_attn = getattr(training_args, 'enable_mixed_attn', False)
    
    lora_enable = getattr(training_args, 'lora_enable', False)
    key_vis_method = getattr(training_args, 'key_vis_token_method', 'attn_topk')
    topk_attn_layer_idx = getattr(training_args, 'topk_attn_layer_idx', 0)
    
    # Determine attention implementation based on distill and enable_mixed_attn
    # - enable_mixed_attn=True: flash + partial eager (for distillation with speedup)
    # - enable_mixed_attn=False + distill=1: eager (need attention weights for distillation)
    # - enable_mixed_attn=False + distill=0: flash (fastest, no distillation)
    if enable_mixed_attn:
        attn_impl = "flash_attention_2"  # will be overridden by mixed attn logic
    elif training_args.distill == 1:
        attn_impl = "eager"
    else:
        attn_impl = "flash_attention_2"
    
    student = load_qwen3_vl_model(
        model_args.model_name_or_path,
        dtype=compute_dtype,
        device="cuda",
        enable_mixed_attn=enable_mixed_attn,
        topk_attn_layer_idx=topk_attn_layer_idx,
        attn_implementation=attn_impl,
    )
    
    if lora_enable:
        from peft import LoraConfig, get_peft_model
        
        target_modules = training_args.lora_target_modules.split(',') if training_args.lora_target_modules else None
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        student.model = get_peft_model(student.model, lora_config)
        
        # Enable input gradients for gradient checkpointing compatibility
        if training_args.gradient_checkpointing:
            if hasattr(student, "enable_input_require_grads"):
                student.enable_input_require_grads()
            elif hasattr(student.model, "enable_input_require_grads"):
                student.model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)
                student.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        
        student._register_vision_hook()
    
    # Freeze ViT, keep merger trainable so L_v_focus gradients reach inference-time parameters
    vision_tower = student.get_vision_tower()
    if vision_tower is not None:
        vision_tower.requires_grad_(False)
        if hasattr(vision_tower, 'merger'):
            vision_tower.merger.requires_grad_(True)
            # Disable gradient checkpointing on the vision encoder: unfreezing the merger
            # causes backward to propagate into the vision encoder, which triggers GC
            # recomputation of all ViT blocks. With frozen ViT params, no GC is needed.
            if hasattr(vision_tower, 'gradient_checkpointing_disable'):
                vision_tower.gradient_checkpointing_disable()
            else:
                vision_tower.gradient_checkpointing = False
    
    teacher = None
    if training_args.distill == 1:
        if model_args.teacher_model_path is None:
            raise ValueError("Teacher model path must be specified when distillation is enabled")
        
        teacher_device = f"cuda:{local_rank}" if local_rank >= 0 else "cuda"
        
        if key_vis_method in ('cosine_l2', 'kve_topk'):
            # cosine_l2 and kve_topk both require all-layer attention weights and value states
            teacher_attn_impl = "eager"
            teacher_enable_mixed = False
        else:
            teacher_attn_impl = attn_impl
            teacher_enable_mixed = enable_mixed_attn
        
        teacher = load_qwen3_vl_model(
            model_args.teacher_model_path,
            dtype=compute_dtype,
            device=teacher_device,
            use_device_map=False,
            enable_mixed_attn=teacher_enable_mixed,
            topk_attn_layer_idx=topk_attn_layer_idx,
            attn_implementation=teacher_attn_impl,
        )
        teacher.to(teacher_device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
    
    # Finetune keeps vision-side parameters trainable; drop text-only samples to
    # avoid rank-dependent unused visual parameters under DDP.
    require_image = training_args.task == "finetune"
    data_module = make_qwen3_vl_data_module(
        processor=student.processor,
        data_path=data_args.data_path,
        image_folder=data_args.image_folder,
        model_max_length=training_args.model_max_length,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
        require_image=require_image,
    )
    
    trainer = VLMTrainer(
        model=student,
        teacher=teacher,
        args=training_args,
        tokenizer=student.processor.tokenizer,
        **data_module
    )
    
    checkpoint_dirs = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoint_dirs:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    
    rank0_print("Saving model...")
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        if lora_enable:
            student.model.save_pretrained(training_args.output_dir)
            student.processor.save_pretrained(training_args.output_dir)
            rank0_print(f"Saved LoRA adapter to {training_args.output_dir}")
        else:
            student.model.save_pretrained(training_args.output_dir)
            student.processor.save_pretrained(training_args.output_dir)
        
        if training_args.distill == 1:
            proj_adapter_path = os.path.join(training_args.output_dir, 'proj_adapter.pt')
            attn_adapter_path = os.path.join(training_args.output_dir, 'attn_adapter.pt')
            
            proj_adapter_state = {name: param.detach().cpu() 
                                  for name, param in trainer._original_proj_adapter.named_parameters()}
            attn_adapter_state = {name: param.detach().cpu() 
                                  for name, param in trainer._original_attn_adapter.named_parameters()}
            
            torch.save(proj_adapter_state, proj_adapter_path)
            torch.save(attn_adapter_state, attn_adapter_path)
            print(f"Saved proj_adapter to {proj_adapter_path}")
            print(f"Saved attn_adapter to {attn_adapter_path}")
    
    rank0_print("Training completed!")


if __name__ == "__main__":
    train()
