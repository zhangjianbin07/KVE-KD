# KVE-KD

Key Visual Evidence-guided Knowledge Distillation (KVE-KD) dynamically focuses
knowledge transfer on task-relevant visual evidence identified by the teacher
model, without introducing inference-time overhead.

![KVE-KD framework](figures/KVE-KD.png)

## Environment

The tested environment uses Linux, Python 3.10, CUDA 11.8, PyTorch 2.4.1,
torchvision 0.19.1, and FlashAttention 2.7.3.

```bash
conda create -n kve-kd python=3.10 -y
conda activate kve-kd

pip install --no-cache-dir \
  torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu118

pip install --no-cache-dir -r requirements.txt

PY_TAG="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
ABI="$(python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
pip install --no-cache-dir \
  "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.4cxx11abi${ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
```

## Dataset

Set `DATA_PATH` to a directory with the following layout:

```text
/path/to/data/
  pretrain_data/
    pretrain.json
    ...
  finetune_data/
    finetune.json
    ...
  benchmark_data/
    gqa/
    mmbench/
    mme/
    pope/
    sqa/
    textvqa/
```

Training JSON files follow the Qwen3-VL conversational data format and refer
to images relative to their corresponding image folder.

## Training and validation

The following command runs KVE-KD pretraining followed by finetuning:

```bash
STUDENT_MODEL=/path/to/Qwen3-VL-2B-Instruct \
TEACHER_MODEL=/path/to/Qwen3-VL-8B-Instruct \
DATA_PATH=/path/to/data \
OUTPUT_BASE_DIR=/path/to/output \
bash run.sh pretrain-finetune
```

Use `DISTILL=0` for supervised finetuning without a teacher model.

`zero2.json` offloads optimizer states to CPU and is suitable for constrained
GPU memory. `zero2_no_offload.json` is faster when sufficient GPU memory is
available.

To resume a trusted local checkpoint with PyTorch 2.4.1, set
`KVE_KD_ALLOW_UNSAFE_TORCH_LOAD=1`. Never use this option with untrusted files.

## Testing

Run all six supported benchmarks with:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/benchmark.sh \
  /path/to/checkpoint \
  /path/to/data \
  /path/to/evaluation-output \
  qwen3_vl
```

Each directory under `DATA_PATH/benchmark_data/` must contain the annotations,
images, conversion scripts, and official evaluation scripts expected by its
runner in `scripts/benchmark/`.

## Acknowledgements

KVE-KD builds on
[Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and uses data and evaluation
conventions from related open-source vision-language model projects, including
[Align-KD](https://github.com/fqhank/Align-KD). Please follow the licenses and
terms of the original models and datasets.
