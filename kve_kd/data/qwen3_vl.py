import os
import re
import json
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Sequence, Tuple, Any
from dataclasses import dataclass
from torch.utils.data import Dataset
import transformers

from kve_kd.constants import (
    IGNORE_INDEX,
    QWEN3_IMAGE_TOKEN_ID,
    QWEN3_VIDEO_TOKEN_ID,
    QWEN3_VISION_START_ID,
    QWEN3_VISION_END_ID,
    QWEN3_IM_END_ID,
    QWEN3_ASSISTANT_TOKEN_ID,
)


def get_rope_index_3(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids_i in enumerate(total_input_ids):
            input_ids_masked = input_ids_i[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids_masked == QWEN3_VISION_START_ID).squeeze(1)
            vision_tokens = input_ids_masked[vision_start_indices + 1]
            image_nums = (vision_tokens == QWEN3_IMAGE_TOKEN_ID).sum()
            video_nums = (vision_tokens == QWEN3_VIDEO_TOKEN_ID).sum()
            input_tokens = input_ids_masked.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if QWEN3_IMAGE_TOKEN_ID in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(QWEN3_IMAGE_TOKEN_ID, st)
                else:
                    ed_image = len(input_tokens) + 1
                if QWEN3_VIDEO_TOKEN_ID in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(QWEN3_VIDEO_TOKEN_ID, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    pass
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    return messages


def preprocess_qwen_visual(
    sources: List[Dict[str, Any]],
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")
    
    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == QWEN3_ASSISTANT_TOKEN_ID:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != QWEN3_IM_END_ID:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def _sample_has_image(sample: Dict[str, Any]) -> bool:
    images = sample.get("image")
    if isinstance(images, str):
        return bool(images.strip())
    if isinstance(images, (list, tuple)):
        return any(bool(image) for image in images)
    return bool(images)


class Qwen3VLDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        processor,
        image_folder: str,
        model_max_length: int = 2048,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
        require_image: bool = False,
    ):
        super().__init__()
        
        self.processor = processor
        self.image_folder = image_folder
        self.model_max_length = model_max_length
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)
        
        self._update_processor_pixels(min_pixels, max_pixels)
        
        with open(data_path, 'r') as f:
            self.list_data_dict = json.load(f)

        if require_image:
            self.list_data_dict = [
                sample for sample in self.list_data_dict if _sample_has_image(sample)
            ]
            kept_count = len(self.list_data_dict)
            if kept_count == 0:
                raise ValueError(f"No image samples remain after filtering {data_path}")
    
    def _update_processor_pixels(self, min_pixels: int, max_pixels: int):
        ip = self.processor.image_processor
        if hasattr(ip, 'min_pixels'):
            ip.min_pixels = min_pixels
        if hasattr(ip, 'max_pixels'):
            ip.max_pixels = max_pixels
    
    def __len__(self) -> int:
        return len(self.list_data_dict)
    
    @property
    def lengths(self) -> List[int]:
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            text_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            length_list.append(text_len + img_tokens)
        return length_list
    
    @property
    def modality_lengths(self) -> List[int]:
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        num_retries = 3
        
        for attempt_idx in range(num_retries):
            try:
                item = self.list_data_dict[i]
                return self._get_item(item)
            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}: {e}")
                if attempt_idx == num_retries - 1:
                    return self._get_empty_sample(i)
        
        return self._get_empty_sample(i)
    
    def _get_empty_sample(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.zeros(1, 1, dtype=torch.long),
            'labels': torch.full((1, 1), IGNORE_INDEX, dtype=torch.long),
            'position_ids': torch.zeros(3, 1, 1, dtype=torch.long),
            'attention_mask': [1],
            'idx': str(i),
            'is_empty': True
        }
    
    def _get_item(self, item: Dict) -> Dict[str, torch.Tensor]:
        source = dict(
            conversations=item["conversations"],
            image=item.get("image", None),
            data_path=self.image_folder,
        )
        if source["image"] is None:
            source.pop("image")
        
        data_dict = preprocess_qwen_visual(
            [source],
            self.processor,
        )

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, (list, tuple)):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        position_ids, _ = get_rope_index_3(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=None,
            second_per_grid_ts=None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        data_dict["idx"] = item.get("id", "")
        data_dict["is_empty"] = False

        return data_dict


def pad_and_cat(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class Qwen3VLDataCollator:
    
    tokenizer: transformers.PreTrainedTokenizer
    model_max_length: int = 2048
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        valid_instances = [inst for inst in instances if not inst.get('is_empty', False)]
        if len(valid_instances) == 0:
            valid_instances = [instances[0]]
        
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in valid_instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        
        input_ids = input_ids[:, :self.model_max_length]
        labels = labels[:, :self.model_max_length]
        position_ids = position_ids[:, :, :self.model_max_length]
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            position_ids=position_ids,
        )
        
        images = list(
            instance["pixel_values"]
            for instance in valid_instances
            if "pixel_values" in instance
        )
        
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in valid_instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        
        batch["idx"] = [inst.get("idx", "") for inst in valid_instances]
        
        return batch


def make_qwen3_vl_data_module(
    processor,
    data_path: str,
    image_folder: str,
    model_max_length: int = 2048,
    min_pixels: int = 256 * 28 * 28,
    max_pixels: int = 1280 * 28 * 28,
    require_image: bool = False,
) -> Dict:
    train_dataset = Qwen3VLDataset(
        data_path=data_path,
        processor=processor,
        image_folder=image_folder,
        model_max_length=model_max_length,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        require_image=require_image,
    )
    
    data_collator = Qwen3VLDataCollator(
        tokenizer=processor.tokenizer,
        model_max_length=model_max_length
    )
    
    return {
        'train_dataset': train_dataset,
        'eval_dataset': None,
        'data_collator': data_collator
    }
