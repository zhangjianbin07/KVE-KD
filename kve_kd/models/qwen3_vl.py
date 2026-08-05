import torch
import torch.nn as nn
import copy
from typing import Optional, Tuple
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from kve_kd.constants import (
    QWEN3_IMAGE_TOKEN_ID,
    QWEN3_VIDEO_TOKEN_ID,
    QWEN3_VISION_START_ID,
    QWEN3_VISION_END_ID,
)


class Qwen3VLWrapper(nn.Module):
    
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_implementation: str = "flash_attention_2",
        use_device_map: bool = False,
        enable_mixed_attn: bool = False,
        topk_attn_layer_idx: int = 0,
    ):
        super().__init__()
        
        self._enable_mixed_attn = enable_mixed_attn
        self._topk_attn_layer_idx = topk_attn_layer_idx
        self._actual_topk_layer_idx = None
        self._mixed_attn_active = False
        self._topk_same_as_first = False
        self._cached_topk_attn = None
        self._cached_first_attn = None
        self._uses_flash_attn = False
        
        if enable_mixed_attn:
            actual_attn_impl = "flash_attention_2"
        else:
            actual_attn_impl = attn_implementation
        
        # Track if using flash attention (cannot output attention weights)
        self._uses_flash_attn = actual_attn_impl in ("flash_attention_2", "sdpa")
        
        load_kwargs = {
            "torch_dtype": dtype,
            "attn_implementation": actual_attn_impl,
        }
        if use_device_map:
            load_kwargs["device_map"] = "auto"
        
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                **load_kwargs
            )
        except Exception as e:
            if enable_mixed_attn and "flash" in str(e).lower():
                print(f"[Mixed Attention] flash_attention_2 load failed, trying sdpa: {e}")
                load_kwargs["attn_implementation"] = "sdpa"
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_path,
                    **load_kwargs
                )
            else:
                raise
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(model_path)
        
        if enable_mixed_attn:
            self._setup_layer_eager_attention(0, cache_target='first_attn')
            if topk_attn_layer_idx != 0:
                self._setup_layer_eager_attention(topk_attn_layer_idx, cache_target='topk_attn')
            else:
                self._topk_same_as_first = True
            self._mixed_attn_active = True
        
        self.config = self.model.config
        self.dtype = dtype
        self.device = device
        
        self._cached_pre_layer0 = None
        self._register_vision_hook()
    
    def _get_base_model(self):
        """Get base HF model, compatible with PeftModel wrapping."""
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                return self.model.get_base_model()
        except ImportError:
            pass
        return self.model
    
    def _register_vision_hook(self):
        def _pre_layer0_hook(module, args, kwargs):
            if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].dim() == 3:
                self._cached_pre_layer0 = args[0]
        
        try:
            layers = self._get_layers()
            layers[0].register_forward_pre_hook(_pre_layer0_hook, with_kwargs=True)
        except Exception as e:
            print(f"[Warning] Failed to register vision hook: {e}")
    
    def _get_layers(self):
        """Get language model layers, compatible with PeftModel."""
        base = self._get_base_model()
        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            return base.model.layers
        if hasattr(base, 'language_model') and hasattr(base.language_model, 'layers'):
            return base.language_model.layers
        raise RuntimeError("Cannot locate language model layers")
    
    def _get_layer(self, layer_idx: int):
        layers = self._get_layers()
        num_layers = len(layers)
        
        actual_idx = layer_idx
        if layer_idx < 0:
            actual_idx = num_layers + layer_idx
        
        if actual_idx < 0 or actual_idx >= num_layers:
            raise ValueError(f"Layer index {layer_idx} out of range [0, {num_layers}) or [-{num_layers}, -1]")
        
        return layers[actual_idx], actual_idx
    
    def _setup_layer_eager_attention(self, layer_idx: int, cache_target: str = 'topk_attn'):
        try:
            layer, actual_idx = self._get_layer(layer_idx)
            
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
            else:
                raise RuntimeError(f"Cannot find attention module in layer {actual_idx}")
            
            original_config = attn.config
            layer_config = copy.copy(original_config)
            layer_config._attn_implementation = "eager"
            attn.config = layer_config
            
            self._register_layer_mask_hook(layer)
            self._register_attention_hooks(attn, cache_target)
            
            if cache_target == 'topk_attn':
                self._actual_topk_layer_idx = actual_idx
            
        except Exception as e:
            print(f"[Mixed Attention] Setup layer {layer_idx} failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _register_layer_mask_hook(self, layer):
        wrapper_self = self
        
        def convert_attention_mask_for_eager(module, args, kwargs):
            attention_mask = kwargs.get('attention_mask', None)
            attention_mask_in_kwargs = 'attention_mask' in kwargs
            if attention_mask is None and len(args) > 1:
                attention_mask = args[1]
            
            hidden_states = kwargs.get('hidden_states', None)
            if hidden_states is None and len(args) > 0:
                hidden_states = args[0]
            
            if hidden_states is None:
                return args, kwargs
            
            batch_size = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            dtype = wrapper_self.dtype
            device = hidden_states.device
            
            def create_causal_mask(b, s, dt, dev):
                causal = torch.triu(
                    torch.ones(s, s, dtype=dt, device=dev),
                    diagonal=1
                )
                causal = causal.masked_fill(causal == 1, torch.finfo(dt).min)
                return causal.unsqueeze(0).unsqueeze(0).expand(b, 1, s, s)
            
            causal_mask = None
            
            if attention_mask is None:
                causal_mask = create_causal_mask(batch_size, seq_len, dtype, device)
            
            elif attention_mask.dim() == 2:
                causal_mask = create_causal_mask(batch_size, seq_len, dtype, device)
                expanded_mask = attention_mask[:, None, None, :].to(dtype).to(device)
                expanded_mask = expanded_mask.expand(batch_size, 1, seq_len, seq_len)
                inverted_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
                causal_mask = causal_mask + inverted_mask
            
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    causal_mask = torch.where(
                        attention_mask,
                        torch.tensor(0.0, dtype=dtype, device=device),
                        torch.tensor(torch.finfo(dtype).min, dtype=dtype, device=device)
                    )
                elif attention_mask.dtype != dtype:
                    causal_mask = attention_mask.to(dtype)
                else:
                    causal_mask = attention_mask
            
            if causal_mask is not None:
                if attention_mask_in_kwargs or attention_mask is None:
                    kwargs['attention_mask'] = causal_mask
                elif len(args) > 1:
                    args = list(args)
                    args[1] = causal_mask
                    args = tuple(args)
            
            return args, kwargs
        
        try:
            layer.register_forward_pre_hook(convert_attention_mask_for_eager, with_kwargs=True)
        except TypeError:
            pass
    
    def _register_attention_hooks(self, attn, cache_target: str = 'topk_attn'):
        wrapper_self = self
        
        def force_output_attentions(module, args, kwargs):
            kwargs['output_attentions'] = True
            return args, kwargs
        
        def capture_attention_weights(module, args, kwargs, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    if cache_target == 'topk_attn':
                        wrapper_self._cached_topk_attn = attn_weights.detach()
                    else:
                        wrapper_self._cached_first_attn = attn_weights.detach()
            return output
        
        try:
            attn.register_forward_pre_hook(force_output_attentions, with_kwargs=True)
        except TypeError:
            original_forward = attn.forward
            def patched_forward(*args, **kwargs):
                kwargs['output_attentions'] = True
                return original_forward(*args, **kwargs)
            attn.forward = patched_forward
        
        try:
            attn.register_forward_hook(capture_attention_weights, with_kwargs=True)
        except TypeError:
            def capture_attention_weights_legacy(module, input, output):
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_weights = output[1]
                    if attn_weights is not None:
                        if cache_target == 'topk_attn':
                            wrapper_self._cached_topk_attn = attn_weights.detach()
                        else:
                            wrapper_self._cached_first_attn = attn_weights.detach()
                return output
            attn.register_forward_hook(capture_attention_weights_legacy)
    
    @property
    def hidden_size(self) -> int:
        return self.config.text_config.hidden_size
    
    @property
    def num_attention_heads(self) -> int:
        return self.config.text_config.num_attention_heads
    
    @property
    def num_hidden_layers(self) -> int:
        return self.config.text_config.num_hidden_layers
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: bool = True,
        output_hidden_states: bool = False,
        use_cache: bool = False,
    ) -> Tuple:
        if self._mixed_attn_active:
            self._cached_topk_attn = None
            self._cached_first_attn = None
        
        if self._mixed_attn_active:
            # Mixed mode: attention weights captured via hooks, don't request globally
            request_attentions = False
        elif self._uses_flash_attn:
            # Pure flash attention: cannot output attention weights, avoid warning
            request_attentions = False
        else:
            # Pure eager attention: can output attention weights
            request_attentions = output_attentions
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            labels=labels,
            output_attentions=request_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=True
        )
        
        v_feature = self._extract_vision_features(input_ids, attention_mask)
        
        if self._mixed_attn_active:
            first_attn = self._cached_first_attn
            if self._topk_same_as_first:
                topk_attn = first_attn
            else:
                topk_attn = self._cached_topk_attn
        else:
            first_attn = None
            topk_attn = None
            if outputs.attentions:
                first_attn = outputs.attentions[0]
                
                layer_idx = self._topk_attn_layer_idx
                num_layers = len(outputs.attentions)
                if layer_idx < 0:
                    layer_idx = num_layers + layer_idx
                if 0 <= layer_idx < num_layers:
                    topk_attn = outputs.attentions[layer_idx]
        
        t_v_mask = None
        if topk_attn is not None:
            t_v_mask = self._build_tv_mask(input_ids, topk_attn, attention_mask)
        
        return outputs, v_feature, topk_attn, first_attn, t_v_mask, None
    
    def _extract_vision_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        pre = getattr(self, "_cached_pre_layer0", None)
        if pre is None:
            raise RuntimeError("pre-layer0 hidden_states not captured")
        
        B, S, D = pre.shape
        feats = []
        max_vis = 0
        
        for b in range(B):
            ids = input_ids[b]
            vision_mask = torch.zeros(S, dtype=torch.bool, device=ids.device)
            starts = (ids == QWEN3_VISION_START_ID).nonzero(as_tuple=True)[0]
            ends = (ids == QWEN3_VISION_END_ID).nonzero(as_tuple=True)[0]
            
            for i in range(min(len(starts), len(ends))):
                st = int(starts[i].item())
                ed = int(ends[i].item())
                if st < ed:
                    vision_mask[st + 1 : ed] = True
            
            if not vision_mask.any():
                vision_mask = (ids == QWEN3_IMAGE_TOKEN_ID) | (ids == QWEN3_VIDEO_TOKEN_ID)
            
            if attention_mask is not None:
                vision_mask = vision_mask & attention_mask[b].bool()
            
            vision_mask = vision_mask.to(pre.device)
            v = pre[b, vision_mask]
            feats.append(v)
            max_vis = max(max_vis, v.shape[0])
        
        if max_vis == 0:
            return pre.new_zeros((B, 1, D))
        
        v_feature = pre.new_zeros((B, max_vis, D))
        for b, v in enumerate(feats):
            if v.shape[0] > 0:
                v_feature[b, : v.shape[0]] = v
        
        return v_feature
    
    def _build_tv_mask(
        self,
        input_ids: torch.LongTensor,
        attention: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.FloatTensor:
        batch_size, seq_len = input_ids.shape
        num_heads = attention.shape[1]
        
        t_v_mask = torch.zeros(
            batch_size, num_heads, seq_len, seq_len,
            dtype=attention.dtype,
            device=attention.device
        )
        
        target_device = attention.device
        
        for b in range(batch_size):
            ids = input_ids[b]
            
            vision_start_pos = (ids == QWEN3_VISION_START_ID).nonzero(as_tuple=True)[0]
            vision_end_pos = (ids == QWEN3_VISION_END_ID).nonzero(as_tuple=True)[0]
            
            vision_mask = torch.zeros(seq_len, dtype=torch.bool, device=ids.device)
            
            if len(vision_start_pos) > 0 and len(vision_end_pos) > 0:
                num_pairs = min(len(vision_start_pos), len(vision_end_pos))
                for i in range(num_pairs):
                    start = vision_start_pos[i].item()
                    end = vision_end_pos[i].item()
                    if start < end:
                        vision_mask[start+1:end] = True
            else:
                vision_mask = (ids == QWEN3_IMAGE_TOKEN_ID) | (ids == QWEN3_VIDEO_TOKEN_ID)
            
            if not vision_mask.any():
                continue
            
            if attention_mask is not None:
                valid_mask = attention_mask[b].bool()
                text_mask = valid_mask & (~vision_mask)
            else:
                text_mask = ~vision_mask
            
            vision_mask = vision_mask.to(target_device)
            text_mask = text_mask.to(target_device)
            
            tv_block = text_mask.unsqueeze(-1) * vision_mask.unsqueeze(0)
            t_v_mask[b] = tv_block.unsqueeze(0).repeat(num_heads, 1, 1)
        
        return t_v_mask
    
    def get_vision_tower(self):
        """Get vision tower, compatible with PeftModel."""
        base = self._get_base_model()
        return getattr(base, 'visual', None)
    
    def eval(self):
        self.model.eval()
        return self
    
    def train(self, mode: bool = True):
        self.model.train(mode)
        return self
    
    def parameters(self, recurse: bool = True):
        return super().parameters(recurse=recurse)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return super().named_parameters(prefix=prefix, recurse=recurse)
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        self.model.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()


def load_qwen3_vl_model(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    use_device_map: bool = False,
    enable_mixed_attn: bool = False,
    topk_attn_layer_idx: int = 0,
    attn_implementation: str = "flash_attention_2",
) -> Qwen3VLWrapper:
    return Qwen3VLWrapper(
        model_path, 
        dtype, 
        device,
        attn_implementation=attn_implementation,
        use_device_map=use_device_map,
        enable_mixed_attn=enable_mixed_attn,
        topk_attn_layer_idx=topk_attn_layer_idx,
    )
