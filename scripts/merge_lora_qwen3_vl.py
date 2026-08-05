#!/usr/bin/env python
"""
Merge LoRA weights for Qwen3-VL models.

Usage:
    python scripts/merge_lora_qwen3_vl.py <base_model> <lora_path> <save_path>

Example:
    python scripts/merge_lora_qwen3_vl.py \
        /path/to/Qwen3-VL-2B-Instruct \
        /path/to/lora_output \
        /path/to/merged_model
"""
import os
import sys
import gc
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))


def log(msg: str):
    print(msg, flush=True)


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB total")


def merge_lora_qwen3_vl(model_base: str, lora_path: str, save_path: str):
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    log("Clearing GPU memory before loading model...")
    clear_gpu_memory()
    
    log(f"Loading base model from: {model_base}")
    log("This may take several minutes for large models...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_base,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    log("Base model loaded successfully")
    
    log(f"Loading processor from: {model_base}")
    processor = AutoProcessor.from_pretrained(model_base)
    log("Processor loaded successfully")
    
    log(f"Loading LoRA adapter from: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    log("LoRA adapter loaded successfully")
    
    log("Merging LoRA weights into base model...")
    model = model.merge_and_unload()
    log("LoRA weights merged successfully")
    
    os.makedirs(save_path, exist_ok=True)
    log(f"Saving merged model to: {save_path}")
    model.save_pretrained(save_path)
    log("Model saved successfully")
    
    log("Saving processor...")
    processor.save_pretrained(save_path)
    log("Processor saved successfully")
    
    log("Merge completed!")


def main():
    if len(sys.argv) != 4:
        print("Usage: python merge_lora_qwen3_vl.py <base_model> <lora_path> <save_path>")
        print("")
        print("Arguments:")
        print("  base_model  - Path to base Qwen3-VL model")
        print("  lora_path   - Path to LoRA adapter output directory")
        print("  save_path   - Path to save merged model")
        sys.exit(1)
    
    merge_lora_qwen3_vl(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()

