import os
import torch
import torch.nn.functional as F
import gc
import numpy as np
import math
from tqdm import tqdm

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale
import comfy.model_base
import comfy.latent_formats
from comfy.cli_args import args, LatentPreviewMethod

def filter_state_dict_by_blocks(state_dict, blocks_mapping):
    filtered_dict = {}

    for key in state_dict:
        if 'transformer_blocks.' in key or 'single_transformer_blocks.' in key:
            block_pattern = key.split('transformer.')[1].split('.', 2)[0:2]
            block_key = f'{block_pattern[0]}.{block_pattern[1]}.'

            if block_key in blocks_mapping:
                filtered_dict[key] = state_dict[key]

    return filtered_dict

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        # Diffusers format
        if k.startswith('transformer.'):
            k = k.replace('transformer.', '')
        if "img_attn.proj" in k:
            k = k.replace("img_attn.proj", "img_attn_proj")
        if "img_attn.qkv" in k:
            k = k.replace("img_attn.qkv", "img_attn_qkv")
        if "txt_attn.proj" in k:
            k = k.replace("txt_attn.proj ", "txt_attn_proj")
        if "txt_attn.qkv" in k:
            k = k.replace("txt_attn.qkv", "txt_attn_qkv")
        new_sd[k] = v
    return new_sd

def apply_lora_weights_manually(transformer, lora_sd, lora_strength):
    # 手动应用LoRA权重，不依赖ComfyUI的load_lora_for_models
    # transformer: 原始模型
    # lora_sd: LoRA权重state_dict
    # lora_strength: LoRA强度
    
    matched_count = 0  # 匹配成功的参数计数
    total_pairs = 0    # 找到的LoRA对总数
    
    # 输出LoRA文件中前10个键，方便分析格式
    print(f"LoRA文件包含 {len(lora_sd)} 个权重键")
    key_samples = list(lora_sd.keys())[:10]
    print(f"LoRA键名示例: {', '.join(key_samples)}")
    
    # 检测不同的LoRA格式
    lora_format = "unknown"
    if any(".lora_up.weight" in k for k in lora_sd.keys()):
        lora_format = "standard"  # 标准格式：key.lora_up.weight/key.lora_down.weight
    elif any(".weight" in k and ".alpha" in k for k in lora_sd.keys()):
        lora_format = "kohya"     # Kohya格式：key.weight/key.alpha
    elif any("_lora_a.weight" in k for k in lora_sd.keys()):
        lora_format = "diffusers" # Diffusers格式：key_lora_a.weight/key_lora_b.weight
    elif any(".lora_A.weight" in k for k in lora_sd.keys()):
        lora_format = "lycoris"   # LyCORIS格式：key.lora_A.weight/key.lora_B.weight
    
    print(f"检测到LoRA格式: {lora_format}")
    
    # 用于记录键对的字典
    up_down_pairs = {}
    
    # 根据不同格式找出所有up-down对
    if lora_format == "standard":
        # 标准格式: key.lora_up.weight/key.lora_down.weight
        for lora_key in lora_sd.keys():
            if ".lora_up.weight" in lora_key:
                base_key = lora_key.replace(".lora_up.weight", "")
                down_key = base_key + ".lora_down.weight"
                
                if down_key in lora_sd:
                    # 提取目标参数名 - 尝试多种格式
                    # 1. 完整路径最后一部分
                    target_key_simple = base_key.split(".")[-1]
                    # 2. 最后两部分 (如 self_attn.v_proj)
                    target_key_compound = '.'.join(base_key.split(".")[-2:]) if len(base_key.split(".")) >= 2 else target_key_simple
                    
                    up_down_pairs[target_key_simple] = (lora_key, down_key)
                    if target_key_simple != target_key_compound:
                        up_down_pairs[target_key_compound] = (lora_key, down_key)
                    total_pairs += 1
                    
    elif lora_format == "kohya":
        # Kohya格式: key.weight/key.alpha
        weights = {}
        alphas = {}
        
        for k in lora_sd.keys():
            if ".weight" in k and not ".alpha" in k:
                weights[k.replace(".weight", "")] = k
            elif ".alpha" in k:
                alphas[k.replace(".alpha", "")] = k
        
        for base in weights.keys():
            if base in alphas:
                up_key = weights[base]
                alpha_key = alphas[base]
                # 在Kohya格式中，同一个key包含了up和down权重
                target_key_simple = base.split(".")[-1]
                target_key_compound = '.'.join(base.split(".")[-2:]) if len(base.split(".")) >= 2 else target_key_simple
                
                # 存储(up键, down键)tuple，在Kohya格式中下两个值相同
                up_down_pairs[target_key_simple] = (up_key, up_key)
                if target_key_simple != target_key_compound:
                    up_down_pairs[target_key_compound] = (up_key, up_key)
                total_pairs += 1
                
    elif lora_format == "diffusers":
        # Diffusers格式: key_lora_a.weight/key_lora_b.weight
        for lora_key in lora_sd.keys():
            if "_lora_a.weight" in lora_key:
                base_key = lora_key.replace("_lora_a.weight", "")
                down_key = base_key + "_lora_b.weight"
                
                if down_key in lora_sd:
                    target_key_simple = base_key.split(".")[-1]
                    target_key_compound = '.'.join(base_key.split(".")[-2:]) if len(base_key.split(".")) >= 2 else target_key_simple
                    
                    up_down_pairs[target_key_simple] = (lora_key, down_key)
                    if target_key_simple != target_key_compound:
                        up_down_pairs[target_key_compound] = (lora_key, down_key)
                    total_pairs += 1
    elif lora_format == "lycoris":
        # LyCORIS格式: key.lora_A.weight/key.lora_B.weight
        for lora_key in lora_sd.keys():
            if ".lora_A.weight" in lora_key:
                base_key = lora_key.replace(".lora_A.weight", "")
                down_key = base_key + ".lora_B.weight"
                
                if down_key in lora_sd:
                    # 尝试多种提取目标键的方式
                    parts = base_key.split(".")
                    
                    # 尝试提取目标模块名
                    target_keys = []
                    # 1. 只取最后一部分（例如img_mlp.fc1的fc1）
                    target_keys.append(parts[-1])
                    # 2. 取最后两部分（例如img_mlp.fc1）
                    if len(parts) >= 2:
                        target_keys.append(f"{parts[-2]}.{parts[-1]}")
                        # 3. 也可能是分开的两部分
                        target_keys.append(parts[-2])
                    # 4. 如果是double_blocks或single_transformer_blocks格式，取出相关部分
                    if "double_blocks" in base_key or "transformer_blocks" in base_key or "single_transformer_blocks" in base_key:
                        for i in range(len(parts)):
                            if parts[i] in ["double_blocks", "transformer_blocks", "single_transformer_blocks"] and i+2 < len(parts):
                                # 尝试用block_idx.module_name格式
                                target_keys.append(f"{parts[i+1]}.{parts[i+2]}")
                                # 单独的模块名
                                target_keys.append(parts[i+2])
                    
                    # 将所有可能的目标键添加到字典
                    for key in set(target_keys):  # 使用set去除重复
                        up_down_pairs[key] = (lora_key, down_key)
                    
                    total_pairs += 1
    else:
        # 尝试直接匹配模型的权重键
        print("未知LoRA格式，尝试直接匹配...")
        for key in lora_sd.keys():
            if ".weight" in key:
                base_key = key.replace(".weight", "")
                # 提取多种可能的目标键
                parts = base_key.split(".")
                if len(parts) > 0:
                    target_key_simple = parts[-1]
                    up_down_pairs[target_key_simple] = (key, key)  # 同一个键作为up和down
                    # 如果有两部分以上，尝试当做复合名称
                    if len(parts) >= 2:
                        target_key_compound = f"{parts[-2]}.{parts[-1]}"
                        up_down_pairs[target_key_compound] = (key, key)
                total_pairs += 1
    
    print(f"找到 {len(up_down_pairs)} 个有效的LoRA up-down对")
    
    # 收集模型中的所有可能目标
    model_module_names = set()
    model_module_paths = {}
    
    def collect_module_names(module, name_prefix=""):
        for name, child in module.named_children():
            full_name = name_prefix + "." + name if name_prefix else name
            
            # 添加各种可能的模块名称形式
            model_module_names.add(name)  # 简单名称
            model_module_paths[name] = full_name  # 记录完整路径
            
            # 复合名称 (例如: parent.child)
            if len(name_prefix.split(".")) > 0:
                parent = name_prefix.split(".")[-1]
                compound_name = f"{parent}.{name}"
                model_module_names.add(compound_name)
                model_module_paths[compound_name] = full_name
                
                # 如果是transformer块，添加特殊匹配
                if ("double_blocks" in name_prefix or "transformer_blocks" in name_prefix or 
                    "single_transformer_blocks" in name_prefix):
                    # 提取块索引
                    parts = name_prefix.split(".")
                    for i, part in enumerate(parts):
                        if part in ["double_blocks", "transformer_blocks", "single_transformer_blocks"] and i+1 < len(parts):
                            block_idx = parts[i+1]
                            # 块索引.module格式
                            block_module = f"{block_idx}.{name}"
                            model_module_names.add(block_module)
                            model_module_paths[block_module] = full_name
                            
            # 递归处理子模块
            collect_module_names(child, full_name)
    
    collect_module_names(transformer)
    print(f"模型中有 {len(model_module_names)} 个可能的模块名称")
    print(f"模型模块示例: {', '.join(sorted(list(model_module_names))[:5])}{'...' if len(model_module_names) > 5 else ''}")
    
    # 查找LoRA键与模型模块名称的交集
    common_keys = set(up_down_pairs.keys()).intersection(model_module_names)
    print(f"找到 {len(common_keys)} 个可能匹配的模块名称")
    if common_keys:
        print(f"匹配的模块名称: {', '.join(sorted(list(common_keys))[:5])}{'...' if len(common_keys) > 5 else ''}")
        
    # 打印LoRA中的一些键来帮助诊断
    print(f"未匹配的LoRA键的示例: {', '.join(list(up_down_pairs.keys())[:5])}")
    
    # 如果没有匹配，尝试更宽松的匹配方式
    if len(common_keys) == 0 and lora_format == "lycoris":
        print("没有找到精确匹配，尝试部分字符串匹配...")
        # 创建一个映射来存储模糊匹配
        fuzzy_matches = {}
        
        for lora_key in up_down_pairs.keys():
            best_match = None
            best_score = 0
            
            for model_key in model_module_names:
                # 基本匹配: 如果一个是另一个的子字符串
                if lora_key in model_key or model_key in lora_key:
                    score = len(set(lora_key).intersection(set(model_key)))
                    if score > best_score:
                        best_score = score
                        best_match = model_key
            
            if best_match and best_score > len(lora_key) / 2:  # 至少半数字符匹配
                fuzzy_matches[lora_key] = best_match
        
        if fuzzy_matches:
            print(f"模糊匹配到 {len(fuzzy_matches)} 个可能的模块")
            for lora_key, model_key in list(fuzzy_matches.items())[:5]:
                print(f"  LoRA键: '{lora_key}' 匹配到模型模块: '{model_key}'")
            
            # 将模糊匹配添加到common_keys
            for lora_key, model_key in fuzzy_matches.items():
                common_keys.add(model_key)
    
    # 递归遍历模型的所有参数
    applied_modules = set()  # 记录已经应用过LoRA的模块路径
    
    # 查找直接匹配的模块路径，获取直接的参数引用
    for model_key in common_keys:
        if model_key in model_module_paths:
            full_path = model_module_paths[model_key]
            parts = full_path.split('.')
            
            # 试图获取指定路径的模块
            current = transformer
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    current = None
                    break
            
            if current is not None and hasattr(current, 'weight'):
                # 直接获得了模块，并且有weight参数
                param = current.weight
                up_key, down_key = up_down_pairs[model_key]
                up_weight = lora_sd[up_key]
                down_weight = lora_sd[down_key]
                
                # 当使用LyCORIS格式时，需要尝试不同的矩阵计算
                try:
                    if lora_format == "lycoris":
                        # LyCORIS格式通常的A/B矩阵需要特殊处理
                        print(f"LyCORIS格式: 尝试应用到 {full_path}.weight")
                        print(f"  参数形状: {param.shape}, A形状: {up_weight.shape}, B形状: {down_weight.shape}")
                        
                        # 由于LoRA格式变化，尝试多种矩阵乘法
                        delta = None
                        
                        # 1. 尝试标准LoRA方式：B x A
                        if down_weight.shape[1] == up_weight.shape[0]:
                            delta = torch.mm(down_weight, up_weight) * lora_strength
                            print("  使用标准LoRA矩阵乘法: B x A")
                            
                        # 2. 尝试反转方式：A x B
                        elif up_weight.shape[1] == down_weight.shape[0]:
                            delta = torch.mm(up_weight, down_weight) * lora_strength
                            print("  使用反转矩阵乘法: A x B")
                            
                        # 3. 尝试转置矩阵
                        elif up_weight.shape[0] == down_weight.shape[0]:
                            delta = torch.mm(up_weight.t(), down_weight) * lora_strength
                            print("  使用转置矩阵乘法: A^T x B")
                        elif up_weight.shape[1] == down_weight.shape[1]:
                            delta = torch.mm(up_weight, down_weight.t()) * lora_strength
                            print("  使用转置矩阵乘法: A x B^T")
                            
                        if delta is not None:
                            print(f"  计算得到delta形状: {delta.shape}")
                            # 如果形状直接匹配
                            if delta.shape == param.shape:
                                param.data += delta
                                matched_count += 1
                                applied_modules.add(full_path)
                                print(f"  直接应用LoRA到 {full_path}.weight")
                            # 尝试reshape
                            else:
                                try:
                                    delta_reshaped = delta.reshape(param.shape)
                                    param.data += delta_reshaped
                                    matched_count += 1
                                    applied_modules.add(full_path)
                                    print(f"  通过reshape成功应用LoRA到 {full_path}.weight")
                                except Exception as e:
                                    print(f"  无法reshape到参数形状: {str(e)}")
                    else:
                        # 其他LoRA格式的标准处理
                        if up_weight.shape[0] == param.shape[0] and down_weight.shape[1] == param.shape[1]:
                            delta = torch.mm(up_weight, down_weight) * lora_strength
                            param.data += delta
                            matched_count += 1
                            applied_modules.add(full_path)
                            print(f"成功应用LoRA到 {full_path}.weight")
                except Exception as e:
                    print(f"应用LoRA到 {full_path}.weight 时出错: {str(e)}")
    
    # 如果上面的直接方法没有匹配到足够数量的参数，则使用递归遍历方法
    if matched_count < min(10, len(common_keys)):
        print("直接路径方法匹配数量较少，尝试递归遍历方法...")
        
        def apply_lora_to_module(module, name_prefix=""):
            nonlocal matched_count
            
            # 如果当前模块路径已经应用过LoRA，则跳过
            if name_prefix in applied_modules:
                return
                
            for name, child in module.named_children():
                full_name = name_prefix + "." + name if name_prefix else name
                apply_lora_to_module(child, full_name)
            
            for param_name, param in module.named_parameters(recurse=False):
                if param_name != "weight":
                    continue
                    
                # 生成各种可能的模块名称匹配
                module_name = name_prefix.split(".")[-1] if name_prefix else ""
                compound_name = '.'.join(name_prefix.split(".")[-2:]) if len(name_prefix.split(".")) >= 2 else module_name
                
                # 如果在transformer块中，尝试取块索引
                block_idx = None
                for i, part in enumerate(name_prefix.split(".")):
                    if part in ["double_blocks", "transformer_blocks", "single_transformer_blocks"] and i+1 < len(name_prefix.split(".")):
                        block_idx = name_prefix.split(".")[i+1]
                        break
                
                block_module = f"{block_idx}.{module_name}" if block_idx else None
                
                # 尝试不同的匹配方式
                target_key = None
                if module_name in common_keys:
                    target_key = module_name
                elif compound_name in common_keys:
                    target_key = compound_name
                elif block_module and block_module in common_keys:
                    target_key = block_module
                
                if target_key:
                    up_key, down_key = up_down_pairs[target_key]
                    up_weight = lora_sd[up_key]
                    down_weight = lora_sd[down_key]
                    
                    # 尝试不同的矩阵计算方式
                    try:
                        if lora_format == "lycoris":
                            # 尝试多种矩阵乘法
                            delta = None
                            
                            # 1. 标准方式
                            if down_weight.shape[1] == up_weight.shape[0]:
                                delta = torch.mm(down_weight, up_weight) * lora_strength
                            # 2. 反转方式
                            elif up_weight.shape[1] == down_weight.shape[0]:
                                delta = torch.mm(up_weight, down_weight) * lora_strength
                            # 3. 转置矩阵
                            elif up_weight.shape[0] == down_weight.shape[0]:
                                delta = torch.mm(up_weight.t(), down_weight) * lora_strength
                            elif up_weight.shape[1] == down_weight.shape[1]:
                                delta = torch.mm(up_weight, down_weight.t()) * lora_strength
                                
                            if delta is not None:
                                # 如果形状匹配
                                if delta.shape == param.shape:
                                    param.data += delta
                                    matched_count += 1
                                    print(f"成功应用LoRA到 {name_prefix}.{param_name} (匹配键: {target_key})")
                                # 尝试reshape
                                else:
                                    try:
                                        delta_reshaped = delta.reshape(param.shape)
                                        param.data += delta_reshaped
                                        matched_count += 1
                                        print(f"通过reshape成功应用LoRA到 {name_prefix}.{param_name}")
                                    except:
                                        print(f"形状不匹配: {name_prefix}.{param_name} (匹配键: {target_key})")
                                        print(f"  参数形状: {param.shape}, delta形状: {delta.shape}")
                        else:
                            # 其他LoRA格式的标准处理
                            if up_weight.shape[0] == param.shape[0] and down_weight.shape[1] == param.shape[1]:
                                delta = torch.mm(up_weight, down_weight) * lora_strength
                                param.data += delta
                                matched_count += 1
                                print(f"成功应用LoRA到 {name_prefix}.{param_name} (匹配键: {target_key})")
                            else:
                                print(f"形状不匹配: {name_prefix}.{param_name} (匹配键: {target_key})")
                                print(f"  参数形状: {param.shape}, up形状: {up_weight.shape}, down形状: {down_weight.shape}")
                    except Exception as e:
                        print(f"应用LoRA到 {name_prefix}.{param_name} 时出错: {str(e)}")
        
        # 开始递归应用
        apply_lora_to_module(transformer)
    
    print(f"成功应用了 {matched_count}/{total_pairs} 个LoRA权重对")
    return transformer

script_directory = os.path.dirname(os.path.abspath(__file__))
vae_scaling_factor = 0.476986

from .diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from .diffusers_helper.memory import DynamicSwapInstaller, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask
from .diffusers_helper.bucket_tools import find_nearest_bucket

class HyVideoModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}
        self.load_device = mm.get_torch_device()

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v


class HyVideoModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = comfy.latent_formats.HunyuanVideo
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        self.memory_usage_factor = 2.0
        self.unet_config["disable_unet_model_creation"] = True

class FramePackTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_single_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable single block compilation"}),
                "compile_double_blocks": ("BOOLEAN", {"default": True, "tooltip": "Enable double block compilation"}),
            },
        }
    RETURN_TYPES = ("FRAMEPACKCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "HunyuanVideoWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_single_blocks, compile_double_blocks):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_single_blocks": compile_single_blocks,
            "compile_double_blocks": compile_double_blocks
        }

        return (compile_args, )

#region Model loading
class DownloadAndLoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["lllyasviel/FramePackI2V_HY"],),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
                "lora": ("FRAMEPACKLORA", {"default": None}),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(self, model, base_precision, quantization,
                  compile_args=None, attention_mode="sdpa", lora=None):
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        model_path = os.path.join(folder_paths.models_dir, "diffusers", "lllyasviel", "FramePackI2V_HY")
        if not os.path.exists(model_path):
            print(f"Downloading clip model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model,
                local_dir=model_path,
                local_dir_use_symlinks=False,
            )

        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(model_path, torch_dtype=base_dtype, attention_mode=attention_mode).cpu()
        
        # 创建一个ModelPatcher用于应用LoRA
        from comfy import model_patcher
        comfy_model = model_patcher.ModelPatcher(transformer, device, offload_device)
        
        # 加载LoRA
        if lora is not None:
            from comfy.sd import load_lora_for_models
            for l in lora:
                print(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                lora_path = l["path"]
                lora_strength = l["strength"]
                lora_sd = load_torch_file(lora_path, safe_load=True)
                lora_sd = standardize_lora_key_format(lora_sd)
                
                if l["blocks"]:
                    lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"])
                
                comfy_model, _ = load_lora_for_models(comfy_model, None, lora_sd, lora_strength, 0)
        
        # 恢复使用transformer对象
        transformer = comfy_model.model
        comfy.model_management.load_models_gpu([comfy_model])
        
        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        if quantization == 'fp8_e4m3fn' or quantization == 'fp8_e4m3fn_fast':
            transformer = transformer.to(torch.float8_e4m3fn)
            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)
        elif quantization == 'fp8_e5m2':
            transformer = transformer.to(torch.float8_e5m2)
        else:
            transformer = transformer.to(base_dtype)

        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
               
            #transformer = torch.compile(transformer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )
    
class LoadFramePackModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
                "lora": ("FRAMEPACKLORA", {"default": None}),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"

    def loadmodel(self, model, base_precision, quantization,
                  compile_args=None, attention_mode="sdpa", lora=None):
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        model_config_path = os.path.join(script_directory, "transformer_config.json")
        import json
        with open(model_config_path, "r") as f:
            config = json.load(f)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        
        with init_empty_weights():
            transformer = HunyuanVideoTransformer3DModelPacked(**config, attention_mode=attention_mode)

        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_scaled":
            dtype = torch.float8_e4m3fn
        elif quantization == "fp8_e5m2":
            dtype = torch.float8_e5m2
        else:
            dtype = base_dtype
        print("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in transformer.named_parameters())
        for name, param in tqdm(transformer.named_parameters(), 
                desc=f"Loading transformer parameters to {offload_device}", 
                total=param_count,
                leave=True):
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
   
            set_module_tensor_to_device(transformer, name, device=offload_device, dtype=dtype_to_use, value=sd[name])

        # 创建一个ModelPatcher用于应用LoRA
        from comfy import model_patcher
        comfy_model = model_patcher.ModelPatcher(transformer, device, offload_device)
        
        # 加载LoRA
        if lora is not None:
            from comfy.sd import load_lora_for_models
            for l in lora:
                print(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                lora_path = l["path"]
                lora_strength = l["strength"]
                lora_sd = load_torch_file(lora_path, safe_load=True)
                lora_sd = standardize_lora_key_format(lora_sd)
                
                if l["blocks"]:
                    lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"])
                
                comfy_model, _ = load_lora_for_models(comfy_model, None, lora_sd, lora_strength, 0)
        
        # 恢复使用transformer对象
        transformer = comfy_model.model
        comfy.model_management.load_models_gpu([comfy_model])

        if quantization == "fp8_e4m3fn_fast":
            from .fp8_optimization import convert_fp8_linear
            convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)
      

        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
               
            #transformer = torch.compile(transformer, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )

class LoadFramePackModelWithLoRA:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn",
                    "sageattn",
                    ], {"default": "sdpa"}),
                "compile_args": ("FRAMEPACKCOMPILEARGS", ),
                "lora": ("FRAMEPACKLORA", {"default": None}),
            }
        }

    RETURN_TYPES = ("FramePackMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "加载FramePack模型并使用手动方法应用LoRA"

    def loadmodel(self, model, base_precision, quantization,
                  compile_args=None, attention_mode="sdpa", lora=None):
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        model_config_path = os.path.join(script_directory, "transformer_config.json")
        import json
        with open(model_config_path, "r") as f:
            config = json.load(f)
        sd = load_torch_file(model_path, device=offload_device, safe_load=True)
        
        with init_empty_weights():
            transformer = HunyuanVideoTransformer3DModelPacked(**config, attention_mode=attention_mode)

        params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
        if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_scaled":
            dtype = torch.float8_e4m3fn
        elif quantization == "fp8_e5m2":
            dtype = torch.float8_e5m2
        else:
            dtype = base_dtype
        print("Using accelerate to load and assign model weights to device...")
        param_count = sum(1 for _ in transformer.named_parameters())
        for name, param in tqdm(transformer.named_parameters(), 
                desc=f"Loading transformer parameters to {offload_device}", 
                total=param_count,
                leave=True):
            dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
   
            set_module_tensor_to_device(transformer, name, device=offload_device, dtype=dtype_to_use, value=sd[name])

        # 使用手动方法应用LoRA，避免使用ModelPatcher
        if lora is not None:
            for l in lora:
                print(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                lora_path = l["path"]
                lora_strength = l["strength"]
                lora_sd = load_torch_file(lora_path, safe_load=True)
                lora_sd = standardize_lora_key_format(lora_sd)
                
                if l["blocks"]:
                    lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"])
                
                # 使用我们的自定义函数手动应用LoRA
                transformer = apply_lora_weights_manually(transformer, lora_sd, lora_strength)

        if quantization == "fp8_e4m3fn_fast":
            from .fp8_optimization import convert_fp8_linear
            convert_fp8_linear(transformer, base_dtype, params_to_keep=params_to_keep)
      

        DynamicSwapInstaller.install_model(transformer, device=device)

        if compile_args is not None:
            if compile_args["compile_single_blocks"]:
                for i, block in enumerate(transformer.single_transformer_blocks):
                    transformer.single_transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
            if compile_args["compile_double_blocks"]:
                for i, block in enumerate(transformer.transformer_blocks):
                    transformer.transformer_blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])

        pipe = {
            "transformer": transformer.eval(),
            "dtype": base_dtype,
        }
        return (pipe, )

class FramePackFindNearestBucket:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Image to resize"}),
            "base_resolution": ("INT", {"default": 640, "min": 64, "max": 2048, "step": 16, "tooltip": "Width of the image to encode"}),
            },
        }

    RETURN_TYPES = ("INT", "INT", )
    RETURN_NAMES = ("width","height",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "Finds the closes resolution bucket as defined in the orignal code"

    def process(self, image, base_resolution):

        H, W = image.shape[1], image.shape[2]

        new_height, new_width = find_nearest_bucket(H, W, resolution=base_resolution)

        return (new_width, new_height, )


class FramePackSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "image_embeds": ("CLIP_VISION_OUTPUT", ),
                "steps": ("INT", {"default": 30, "min": 1}),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "Use teacache for faster sampling."}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for the relative L1 loss."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "latent_window_size": ("INT", {"default": 9, "min": 1, "max": 33, "step": 1, "tooltip": "The size of the latent window to use for sampling."}),
                "total_second_length": ("FLOAT", {"default": 5, "min": 1, "max": 120, "step": 0.1, "tooltip": "The total length of the video in seconds."}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 128.0, "step": 0.1, "tooltip": "The amount of GPU memory to preserve."}),
                "sampler": (["unipc_bh1", "unipc_bh2"],
                    {
                        "default": 'unipc_bh1'
                    }),
            },
            "optional": {
                "start_latent": ("LATENT", {"tooltip": "init Latents to use for image2video"} ),
                "end_latent": ("LATENT", {"tooltip": "end Latents to use for image2video"} ),
                "end_image_embeds": ("CLIP_VISION_OUTPUT", {"tooltip": "end Image's clip embeds"} ),
                "embed_interpolation": (["weighted_average", "linear"], {"default": 'linear', "tooltip": "Image embedding interpolation type. If linear, will smoothly interpolate with time, else it'll be weighted average with the specified weight."}),
                "start_embed_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Weighted average constant for image embed interpolation. If end image is not set, the embed's strength won't be affected"}),
                "initial_samples": ("LATENT", {"tooltip": "init Latents to use for video2video"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"

    def process(self, model, shift, positive, negative, latent_window_size, use_teacache, total_second_length, teacache_rel_l1_thresh, image_embeds, steps, cfg,
                guidance_scale, seed, sampler, gpu_memory_preservation, start_latent=None, end_latent=None, end_image_embeds=None, embed_interpolation="linear", start_embed_strength=1.0, initial_samples=None, denoise_strength=1.0):
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))
        print("total_latent_sections: ", total_latent_sections)

        transformer = model["transformer"]
        base_dtype = model["dtype"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        start_latent = start_latent["samples"] * vae_scaling_factor
        if initial_samples is not None:
            initial_samples = initial_samples["samples"] * vae_scaling_factor
        if end_latent is not None:
            end_latent = end_latent["samples"] * vae_scaling_factor
        has_end_image = end_latent is not None
        print("start_latent", start_latent.shape)
        B, C, T, H, W = start_latent.shape

        start_image_encoder_last_hidden_state = image_embeds["last_hidden_state"].to(base_dtype).to(device)

        if has_end_image:
            assert end_image_embeds is not None
            end_image_encoder_last_hidden_state = end_image_embeds["last_hidden_state"].to(base_dtype).to(device)
        else:
            end_image_encoder_last_hidden_state = torch.zeros_like(start_image_encoder_last_hidden_state)

        llama_vec = positive[0][0].to(base_dtype).to(device)
        clip_l_pooler = positive[0][1]["pooled_output"].to(base_dtype).to(device)

        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(base_dtype)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(base_dtype).to(device)
        else:
            llama_vec_n = torch.zeros_like(llama_vec, device=device)
            clip_l_pooler_n = torch.zeros_like(clip_l_pooler, device=device)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            

        # Sampling

        rnd = torch.Generator("cpu").manual_seed(seed)
        
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, H, W), dtype=torch.float32).cpu()
       
        total_generated_latent_frames = 0

        latent_paddings_list = list(reversed(range(total_latent_sections)))
        latent_paddings = latent_paddings_list.copy()  # Create a copy for iteration

        comfy_model = HyVideoModel(
                HyVideoModelConfig(base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=device,
            )
      
        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, torch.device("cpu"))
        from latent_preview import prepare_callback
        callback = prepare_callback(patcher, steps)

        move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            latent_paddings_list = latent_paddings.copy()

        for i, latent_padding in enumerate(latent_paddings):
            print(f"latent_padding: {latent_padding}")
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

            if embed_interpolation == "linear":
                if total_latent_sections <= 1:
                    frac = 1.0  # Handle case with only one section
                else:
                    frac = 1 - i / (total_latent_sections - 1)  # going backwards
            else:
                frac = start_embed_strength if has_end_image else 1.0

            image_encoder_last_hidden_state = start_image_encoder_last_hidden_state * frac + (1 - frac) * end_image_encoder_last_hidden_state

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            # Use end image latent for the first section if provided
            if has_end_image and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            #vid2vid
            
            if initial_samples is not None:
                total_length = initial_samples.shape[2]
                
                # Get the max padding value for normalization
                max_padding = max(latent_paddings_list)
                
                if is_last_section:
                    # Last section should capture the end of the sequence
                    start_idx = max(0, total_length - latent_window_size)
                else:
                    # Calculate windows that distribute more evenly across the sequence
                    # This normalizes the padding values to create appropriate spacing
                    if max_padding > 0:  # Avoid division by zero
                        progress = (max_padding - latent_padding) / max_padding
                        start_idx = int(progress * max(0, total_length - latent_window_size))
                    else:
                        start_idx = 0
                
                end_idx = min(start_idx + latent_window_size, total_length)
                print(f"start_idx: {start_idx}, end_idx: {end_idx}, total_length: {total_length}")
                input_init_latents = initial_samples[:, :, start_idx:end_idx, :, :].to(device)
          

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_rel_l1_thresh)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=base_dtype, enabled=True):
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler=sampler,
                    initial_latent=input_init_latents if initial_samples is not None else None,
                    strength=denoise_strength,
                    width=W * 8,
                    height=H * 8,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=guidance_scale,
                    guidance_rescale=0,
                    shift=shift if shift != 0 else None,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=device,
                    dtype=base_dtype,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if is_last_section:
                break

        transformer.to(offload_device)
        mm.soft_empty_cache()

        return {"samples": real_history_latents / vae_scaling_factor},
class FramePackLoraBlockEdit:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        arg_dict = {}
        argument = ("BOOLEAN", {"default": True})

        for i in range(20):
            arg_dict[f"transformer_blocks.{i}."] = argument

        for i in range(40):
            arg_dict[f"single_transformer_blocks.{i}."] = argument

        return {"required": arg_dict}

    RETURN_TYPES = ("SELECTEDBLOCKS", )
    RETURN_NAMES = ("blocks", )
    FUNCTION = "select"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "选择要应用LoRA的模型块"

    def select(self, **kwargs):
        selected_blocks = {k: v for k, v in kwargs.items() if v is True}
        print("Selected blocks: ", selected_blocks)
        return (selected_blocks,)

class FramePackLoraSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
               "lora": (folder_paths.get_filename_list("loras"),
                {"tooltip": "LORA模型应位于ComfyUI/models/loras目录，扩展名为.safetensors"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.0001, "tooltip": "LoRA的强度，设置为0可卸载LoRA"}),
            },
            "optional": {
                "prev_lora":("FRAMEPACKLORA", {"default": None, "tooltip": "用于加载多个LoRA"}),
                "blocks":("SELECTEDBLOCKS", ),
            }
        }

    RETURN_TYPES = ("FRAMEPACKLORA",)
    RETURN_NAMES = ("lora", )
    FUNCTION = "getlorapath"
    CATEGORY = "FramePackWrapper"
    DESCRIPTION = "从ComfyUI/models/loras选择LoRA模型"

    def getlorapath(self, lora, strength, blocks=None, prev_lora=None):
        loras_list = []

        lora = {
            "path": folder_paths.get_full_path("loras", lora),
            "strength": strength,
            "name": lora.split(".")[0],
            "blocks": blocks
        }
        if prev_lora is not None:
            loras_list.extend(prev_lora)

        loras_list.append(lora)
        return (loras_list,)
    
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadFramePackModel": DownloadAndLoadFramePackModel,
    "FramePackSampler": FramePackSampler,
    "FramePackTorchCompileSettings": FramePackTorchCompileSettings,
    "FramePackFindNearestBucket": FramePackFindNearestBucket,
    "LoadFramePackModel": LoadFramePackModel,
    "FramePackLoraBlockEdit": FramePackLoraBlockEdit,
    "FramePackLoraSelect": FramePackLoraSelect,
    "LoadFramePackModelWithLoRA": LoadFramePackModelWithLoRA,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadFramePackModel": "(Down)Load FramePackModel",
    "FramePackSampler": "FramePackSampler",
    "FramePackTorchCompileSettings": "Torch Compile Settings",
    "FramePackFindNearestBucket": "Find Nearest Bucket",
    "LoadFramePackModel": "Load FramePackModel",
    "FramePackLoraBlockEdit": "FramePack LoRA Block Edit",
    "FramePackLoraSelect": "FramePack LoRA Select",
    "LoadFramePackModelWithLoRA": "Load FramePackModel With LoRA",
    }

