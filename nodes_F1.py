import os
import torch
import math
import re

import comfy.model_management as mm
import comfy.model_base
import comfy.model_patcher

from .nodes import HyVideoModel, HyVideoModelConfig # Import the classes

script_directory = os.path.dirname(os.path.abspath(__file__))
vae_scaling_factor = 0.476986

from .diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModel
from .diffusers_helper.memory import move_model_to_device_with_memory_preservation
from .diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from .diffusers_helper.utils import crop_or_pad_yield_mask

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union # Add necessary types

from latent_preview import prepare_callback

# --- Helper Classes and Functions for Timestamped Prompts ---
@dataclass
class PromptSection:
    prompt: str
    start_time: float = 0.0  # in seconds
    end_time: Optional[float] = None  # in seconds, None means until the end

def snap_to_section_boundaries(prompt_sections: List[PromptSection], latent_window_size: int, fps: int = 30) -> List[PromptSection]:

    section_frame_duration = latent_window_size * 4 - 3
    if section_frame_duration <= 0: section_frame_duration = 1
    section_duration_sec = section_frame_duration / float(fps)
    if section_duration_sec <= 1e-5: section_duration_sec = 1.0 / fps # Avoid zero or near-zero duration

    aligned_sections = []
    for section in prompt_sections:
        aligned_start = round(section.start_time / section_duration_sec) * section_duration_sec
        aligned_end = None
        if section.end_time is not None:
            aligned_end = round(section.end_time / section_duration_sec) * section_duration_sec
            if aligned_end <= aligned_start + 1e-5: # Ensure minimum duration
                aligned_end = aligned_start + section_duration_sec
        aligned_sections.append(PromptSection(
            prompt=section.prompt,
            start_time=aligned_start,
            end_time=aligned_end
        ))
    return aligned_sections

def parse_timestamped_prompt_f1(prompt_text: str, total_duration: float, latent_window_size: int = 9) -> List[PromptSection]:

    #Parse a prompt with timestamps like [0s: text], [1.5s-3s: text] for F1-style forward generation.
    #Returns a list of PromptSection objects with timestamps aligned to section boundaries.
    sections = []
    # Corrected Regex: Catches [Xs: text] or [Xs-Ys: text]
    timestamp_pattern = r'\[\s*(\d+(?:\.\d+)?s)\s*(?:-\s*(\d+(?:\.\d+)?s)\s*)?:\s*(.*?)\s*\]'
    matches = list(re.finditer(timestamp_pattern, prompt_text))
    last_end_index = 0

    if not matches:
        return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]

    for match in matches:
        plain_text_before = prompt_text[last_end_index:match.start()].strip()
        current_start_time_str = match.group(1)
        current_start_time = float(current_start_time_str.rstrip('s'))
        if plain_text_before:
            previous_end_time = sections[-1].end_time if sections and sections[-1].end_time is not None else (sections[-1].start_time if sections else 0.0)
            if current_start_time > previous_end_time + 1e-5:
                sections.append(PromptSection(prompt=plain_text_before, start_time=previous_end_time, end_time=current_start_time))
            elif not sections and current_start_time > 1e-5: # Plain text at the very beginning
                 sections.append(PromptSection(prompt=plain_text_before, start_time=0.0, end_time=current_start_time))

        end_time_str = match.group(2)
        section_text = match.group(3).strip()
        start_time = current_start_time # Already parsed
        end_time = float(end_time_str.rstrip('s')) if end_time_str else None
        sections.append(PromptSection(prompt=section_text, start_time=start_time, end_time=end_time))
        last_end_index = match.end()

    plain_text_after = prompt_text[last_end_index:].strip()
    if plain_text_after:
         previous_end_time = sections[-1].end_time if sections and sections[-1].end_time is not None else sections[-1].start_time
         if total_duration > previous_end_time + 1e-5:
              sections.append(PromptSection(prompt=plain_text_after, start_time=previous_end_time, end_time=None))

    if not sections: # Should not happen if regex matched, but safety
         return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]

    sections.sort(key=lambda x: x.start_time)

    # Sanitize and Fill Gaps/Set End Times
    sanitized_sections = []
    current_time = 0.0
    for i, section in enumerate(sections):
        section_start = max(current_time, section.start_time) # Ensure monotonic increase
        section_start = min(section_start, total_duration) # Clamp to total duration

        # Fill gap if needed
        if section_start > current_time + 1e-5:
             filler_prompt = sanitized_sections[-1].prompt if sanitized_sections else "" # Use previous prompt
             sanitized_sections.append(PromptSection(prompt=filler_prompt, start_time=current_time, end_time=section_start))

        # Determine end time
        section_end = section.end_time
        if section_end is None:
            if i + 1 < len(sections):
                 next_start = max(section_start, sections[i+1].start_time) # Ensure next start is after current start
                 section_end = min(next_start, total_duration) # End before next or at total duration
            else:
                 section_end = total_duration # Last section ends at total duration
        else:
            section_end = min(max(section_start, section_end), total_duration) # Clamp user-defined end

        # Add the section if it has duration
        if section_end > section_start + 1e-5:
             sanitized_sections.append(PromptSection(prompt=section.prompt, start_time=section_start, end_time=section_end))
             current_time = section_end # Update current time marker
        elif i == len(sections) - 1 and math.isclose(section_start, total_duration): # Allow point at the end? No, remove.
             pass

    if not sanitized_sections:
         return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]

    # Snap timestamps to boundaries
    aligned_sections = snap_to_section_boundaries(sanitized_sections, latent_window_size)

    # Merge identical consecutive prompts after snapping
    merged_sections = []
    if not aligned_sections: return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]

    current_merged = aligned_sections[0]
    for i in range(1, len(aligned_sections)):
        next_sec = aligned_sections[i]
        # Merge if prompts are identical and sections are contiguous (or very close after snapping)
        if next_sec.prompt == current_merged.prompt and abs(next_sec.start_time - current_merged.end_time) < 0.01:
            current_merged.end_time = next_sec.end_time # Extend the end time
        else:
            current_merged.end_time = max(current_merged.start_time, current_merged.end_time)
            if current_merged.start_time < current_merged.end_time - 1e-5:
                 merged_sections.append(current_merged)
            current_merged = next_sec

    current_merged.end_time = max(current_merged.start_time, current_merged.end_time)
    if current_merged.start_time < current_merged.end_time - 1e-5:
        merged_sections.append(current_merged)

    if not merged_sections: return [PromptSection(prompt=prompt_text.strip(), start_time=0.0, end_time=total_duration)]

    print("Parsed Prompt Sections (F1):")
    for sec in merged_sections: print(f"  [{sec.start_time:.3f}s - {sec.end_time:.3f}s]: {sec.prompt}")
    return merged_sections
# --- End Helper Code ---

class FramePackSampler_F1:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("FramePackMODEL",),
                "positive_timed_data": ("TIMED_CONDITIONING_WITH_METADATA", { "tooltip": "Output from FramePackTimestampedTextEncode. Dictionary containing sections, duration, and window size."}),
                "negative": ("CONDITIONING",),
                "steps": ("INT", {"default": 30, "min": 1}),
                "use_teacache": ("BOOLEAN", {"default": True, "tooltip": "Use teacache for faster sampling."}),
                "teacache_rel_l1_thresh": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The threshold for the relative L1 loss."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 32.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "gpu_memory_preservation": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 128.0, "step": 0.1, "tooltip": "The amount of GPU memory to preserve."}),
                "sampler": (["unipc_bh1", "unipc_bh2"],
                    {
                        "default": 'unipc_bh1'
                    }),
            },
            "optional": {
                "start_latent": ("LATENT", {"tooltip": "init Latents to use for image2video"} ),
                "start_image_embeds": ("CLIP_VISION_OUTPUT", ),
                "end_latent": ("LATENT", {"tooltip": "end Latents to use for image2video"} ),
                "end_image_embeds": ("CLIP_VISION_OUTPUT", {"tooltip": "end Image's clip embeds"} ),
                "embed_interpolation": (["disabled", "weighted_average", "linear"], {"default": 'disabled', "tooltip": "Image embedding interpolation type. If linear, will smoothly interpolate with time, else it'll be weighted average with the specified weight."}),
                "start_embed_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Weighted average constant for image embed interpolation. If end image is not set, the embed's strength won't be affected"}),
                "initial_samples": ("LATENT", {"tooltip": "init Latents to use for video2video"} ),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "FramePackWrapper"

    def process(self, model, positive_timed_data, negative, use_teacache, teacache_rel_l1_thresh, steps, cfg,
                guidance_scale, shift, seed, sampler, gpu_memory_preservation, start_image_embeds=None, start_latent=None, end_latent=None, end_image_embeds=None, embed_interpolation="linear", start_embed_strength=1.0, initial_samples=None, denoise_strength=1.0):

        # --- Extract data from positive_timed_data --- 
        positive_timed_list = positive_timed_data["sections"]
        total_second_length = positive_timed_data["total_duration"]
        latent_window_size = positive_timed_data["window_size"]
        prompt_blend_sections = positive_timed_data["blend_sections"]
        print(f"Received - Total Duration: {total_second_length}s, Window Size: {latent_window_size}, Blend Sections: {prompt_blend_sections}")

        # --- F1 Model Type Assumption ---
        # We assume the model loaded into this node is the F1 type.

        # Calculate total sections based on time and window size
        section_frame_duration = latent_window_size * 4 - 3
        if section_frame_duration <= 0: section_frame_duration = 1
        fps = 30 # Assume 30 fps
        section_duration_sec = section_frame_duration / float(fps)
        if section_duration_sec <= 0: section_duration_sec = 1.0 / fps

        # Calculate total sections needed to cover the duration
        total_latent_sections = int(math.ceil(total_second_length / section_duration_sec))
        total_latent_sections = max(total_latent_sections, 1)
        print(f"Total latent sections calculated: {total_latent_sections} (Duration: {total_second_length}s, Section time: {section_duration_sec:.3f}s)")


        transformer = model["transformer"]
        base_dtype = model["dtype"]

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()

        if start_latent is None:
            # Handle case where start_latent is not provided (e.g., create default black latent)
             # Get model's expected channel count (often 16 for FramePack)
             latent_channels = getattr(transformer.config, 'in_channels', 16)
             # Determine a default spatial size if not derivable (e.g., 64x64 or based on bucket?)
             # Using a common default like 64x64 / 8 = 8x8 latent space, but this might need adjustment
             H = W = 64 # Default spatial size assumption
             print(f"Warning: start_latent not provided. Creating default black latent ({latent_channels}x1x{H}x{W}).")
             start_latent_tensor = torch.zeros([1, latent_channels, 1, H, W], dtype=torch.float32)
        else:
            start_latent_tensor = start_latent["samples"] # Get tensor from dictionary

        # Get shape AFTER potentially creating the default
        B, C, T, H, W = start_latent_tensor.shape
        print(f"Latent dimensions: B={B}, C={C}, T={T}, H={H}, W={W}")

        start_latent_tensor = start_latent_tensor * vae_scaling_factor

        if initial_samples is not None:
            initial_samples = initial_samples["samples"] * vae_scaling_factor
        if end_latent is not None:
            end_latent = end_latent["samples"] * vae_scaling_factor
        has_end_image = end_latent is not None

        start_image_encoder_last_hidden_state = None # Initialize to None
        if start_image_embeds is not None:
            start_image_encoder_last_hidden_state = start_image_embeds["last_hidden_state"].to(base_dtype).to(device)

        end_image_encoder_last_hidden_state = None # Initialize to None
        if has_end_image and embed_interpolation != "disabled" and end_image_embeds is not None:
            end_image_encoder_last_hidden_state = end_image_embeds["last_hidden_state"].to(base_dtype).to(device)
        elif start_image_encoder_last_hidden_state is not None: # Only create zeros if start exists
            end_image_encoder_last_hidden_state = torch.zeros_like(start_image_encoder_last_hidden_state)

        # --- Conditioning Setup ---
        # Negative conditioning
        if not math.isclose(cfg, 1.0):
            llama_vec_n = negative[0][0].to(dtype=base_dtype, device=device)
            clip_l_pooler_n = negative[0][1]["pooled_output"].to(dtype=base_dtype, device=device)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
        else:
            # Need dummy tensors with correct shape and device. Use shape from the first positive section.
            if positive_timed_list:
                try:
                    first_pos_cond = positive_timed_list[0][2][0][0].to(device=device)
                    first_pos_pooled = positive_timed_list[0][2][0][1]["pooled_output"].to(device=device)
                    llama_vec_n = torch.zeros_like(first_pos_cond)
                    clip_l_pooler_n = torch.zeros_like(first_pos_pooled)
                    # Still need to pad the zero tensor and get the mask
                    llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
                except Exception as e:
                    print(f"Error accessing positive_timed_list for negative shape when cfg=1.0: {e}. Creating fallback zero tensors.")
                    # Fallback zero tensors if list structure is unexpected or empty
                    llama_vec_n = torch.zeros((B, 512, 4096), dtype=base_dtype, device=device) # Guessing shape based on llama
                    llama_attention_mask_n = torch.ones((B, 512), dtype=torch.long, device=device)
                    clip_l_pooler_n = torch.zeros((B, 1280), dtype=base_dtype, device=device) # Guessing shape based on clip-l
            else:
                 # This case remains the same - if no positive sections, create fallback zeros.
                 print("Warning: positive_timed_list is empty when cfg=1.0. Cannot determine negative shape. Creating fallback zero tensors.")
                 llama_vec_n = torch.zeros((B, 512, 4096), dtype=base_dtype, device=device)
                 llama_attention_mask_n = torch.ones((B, 512), dtype=torch.long, device=device)
                 clip_l_pooler_n = torch.zeros((B, 1280), dtype=base_dtype, device=device)

        # Positive conditioning: Handled inside the loop based on time.
        # --- End Conditioning Setup ---

        # Sampling
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3 # Frames generated per step

        # F1 History Latents Initialization
        history_latents = torch.zeros(size=(B, 16, 16 + 2 + 1, H, W), dtype=torch.float32).cpu()
        # F1: Start with the initial latent frame
        history_latents = torch.cat([start_latent_tensor.to(history_latents)], dim=2)
        total_generated_latent_frames = 1 # F1: Start count at 1, representing the initial frame

        # F1 Latent Paddings (determines number of generation steps)
        latent_paddings = [1] * (total_latent_sections - 1) + [0]
        latent_paddings_list = latent_paddings.copy() # For vid2vid indexing


        comfy_model = HyVideoModel(
                HyVideoModelConfig(base_dtype),
                model_type=comfy.model_base.ModelType.FLOW,
                device=device,
            )

        patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, torch.device("cpu"))
        #from latent_preview import prepare_callback # Moved to top
        callback = prepare_callback(patcher, steps)

        move_model_to_device_with_memory_preservation(transformer, target_device=device, preserved_memory_gb=gpu_memory_preservation)

        for i, latent_padding in enumerate(latent_paddings):
            print(f"Sampling Section {i+1}/{total_latent_sections}, latent_padding: {latent_padding}")
            is_last_section = latent_padding == 0

            # F1 logic doesn't seem to use embed interpolation within the loop
            # image_encoder_last_hidden_state = start_image_encoder_last_hidden_state * start_embed_strength
            # ^-- This logic is removed as F1 doesn't use interpolation per step like the other sampler.
            # We just pass the start_image_encoder_last_hidden_state directly to sample_hunyuan below.
            # Handle case where image_embeds wasn't provided
            current_image_embeds = start_image_encoder_last_hidden_state

            # --- Determine Current Positive Conditioning --- 
            # Calculate current time position based on the *start* of the section being generated
            current_time_position = i * section_duration_sec
            current_time_position = max(0.0, current_time_position)
            print(f"  Current time position: {current_time_position:.3f}s")

            active_section_index = -1
            if not positive_timed_list:
                 print("Error: positive_timed_list is empty! Cannot sample.")
                 # Handle error appropriately - maybe return black frames or raise exception?
                 # Returning empty/zeros for now
                 return {"samples": torch.zeros_like(start_latent_tensor) / vae_scaling_factor},

            for idx, (start_sec, end_sec, _) in enumerate(positive_timed_list):
                # Check if current_time_position falls within [start_sec, end_sec)
                if start_sec <= current_time_position + 1e-4 and current_time_position < end_sec - 1e-4:
                    active_section_index = idx
                    # print(f"  Found active prompt section index: {active_section_index} ({start_sec:.2f}s - {end_sec:.2f}s)")
                    break
            else:
                # If no section matches exactly, check edge cases
                if math.isclose(current_time_position, positive_timed_list[-1][1], abs_tol=1e-4):
                     active_section_index = len(positive_timed_list) - 1
                     # print(f"  Time matches end of last section. Using index: {active_section_index}")
                elif current_time_position >= positive_timed_list[-1][1] - 1e-4:
                     active_section_index = len(positive_timed_list) - 1
                     # print(f"  Time past end of last section. Using index: {active_section_index}")
                elif current_time_position < positive_timed_list[0][0] + 1e-4:
                     active_section_index = 0
                     # print(f"  Time before first section. Using index: 0")
                else: # Final fallback if list exists but no match (should be rare)
                    active_section_index = len(positive_timed_list) - 1
                    print(f"  Warning: No exact time match found, using last section index: {active_section_index}")

            print(f"  Selected active prompt index: {active_section_index}")

            # --- Blending Logic --- 
            blend_alpha = 0.0
            prev_section_idx_for_blend = active_section_index
            next_section_idx_for_blend = active_section_index
            current_active_conditioning_tensor = positive_timed_list[active_section_index][2][0][0]

            # Find the index in the original list corresponding to the *start* of the next *different* conditioning
            next_prompt_change_section_start_index = -1
            next_prompt_change_start_time = -1.0
            for k in range(active_section_index + 1, len(positive_timed_list)):
                # Compare the actual conditioning data (tensors)
                if not torch.equal(positive_timed_list[k][2][0][0], current_active_conditioning_tensor):
                    next_prompt_change_start_time = positive_timed_list[k][0]
                    next_prompt_change_section_start_index = int(round(next_prompt_change_start_time / section_duration_sec))
                    prev_section_idx_for_blend = active_section_index # The prompt active before the change
                    next_section_idx_for_blend = k # The prompt active after the change
                    # print(f"  Next prompt change detected at section index ~{next_prompt_change_section_start_index} (time {next_prompt_change_start_time:.2f}s)")
                    break

            # Check if we are within the blend window leading up to the change
            if prompt_blend_sections > 0 and next_prompt_change_section_start_index != -1:
                blend_start_section_idx = next_prompt_change_section_start_index - prompt_blend_sections
                current_physical_section_idx = i # Use the actual loop iteration index

                if current_physical_section_idx >= blend_start_section_idx and current_physical_section_idx < next_prompt_change_section_start_index:
                    blend_progress = (current_physical_section_idx - blend_start_section_idx + 1) / float(prompt_blend_sections)
                    blend_alpha = max(0.0, min(1.0, blend_progress))
                    print(f"  Blending prompts: Section Index {current_physical_section_idx}, Blend Alpha: {blend_alpha:.3f}")
                # No explicit 'else if >= next_prompt_change...' needed, blend_alpha remains 0 if not in window

            # --- End Blending Logic ---

            # Get the conditioning tensors
            if blend_alpha > 0 and prev_section_idx_for_blend != next_section_idx_for_blend:
                # Ensure indices are valid before accessing
                if 0 <= prev_section_idx_for_blend < len(positive_timed_list) and 0 <= next_section_idx_for_blend < len(positive_timed_list):
                    cond_prev = positive_timed_list[prev_section_idx_for_blend][2][0][0].to(dtype=base_dtype, device=device)
                    pooled_prev = positive_timed_list[prev_section_idx_for_blend][2][0][1]['pooled_output'].to(dtype=base_dtype, device=device)
                    cond_next = positive_timed_list[next_section_idx_for_blend][2][0][0].to(dtype=base_dtype, device=device)
                    pooled_next = positive_timed_list[next_section_idx_for_blend][2][0][1]['pooled_output'].to(dtype=base_dtype, device=device)

                    # Pad tensors before lerp
                    padded_cond_prev, mask_prev = crop_or_pad_yield_mask(cond_prev, length=512)
                    padded_cond_next, mask_next = crop_or_pad_yield_mask(cond_next, length=512)

                    llama_vec = torch.lerp(padded_cond_prev, padded_cond_next, blend_alpha)
                    clip_l_pooler = torch.lerp(pooled_prev, pooled_next, blend_alpha) # Poolers assumed same shape
                    llama_attention_mask = mask_prev # Use mask from the first part of lerp
                else:
                     print(f"Warning: Invalid blend indices ({prev_section_idx_for_blend}, {next_section_idx_for_blend}). Using non-blended active prompt.")
                     # Fallback to non-blended active prompt
                     selected_positive = positive_timed_list[active_section_index][2]
                     llama_vec = selected_positive[0][0].to(dtype=base_dtype, device=device)
                     clip_l_pooler = selected_positive[0][1]['pooled_output'].to(dtype=base_dtype, device=device)
                     llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            else:
                # Use the selected active conditioning directly
                selected_positive = positive_timed_list[active_section_index][2]
                llama_vec = selected_positive[0][0].to(dtype=base_dtype, device=device)
                clip_l_pooler = selected_positive[0][1]['pooled_output'].to(dtype=base_dtype, device=device)
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)

            # --- End Determine Current Positive Conditioning ---

            # F1 Indices Calculation
            effective_window_size = int(latent_window_size)
            indices = torch.arange(0, sum([1, 16, 2, 1, effective_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, effective_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            # F1 Clean Latents Calculation
            required_history_len = 16 + 2 + 1 # Need 19 previous frames
            available_history_len = history_latents.shape[2]

            if available_history_len < required_history_len:
                 print(f"Warning: Not enough history frames ({available_history_len}) for clean latents (needed {required_history_len}). Padding with zeros.")
                 # Pad history_latents at the beginning with zeros to meet required length
                 padding_needed = required_history_len - available_history_len
                 padding_shape = list(history_latents.shape)
                 padding_shape[2] = padding_needed
                 zero_padding = torch.zeros(padding_shape, dtype=history_latents.dtype, device=history_latents.device)
                 padded_history = torch.cat([zero_padding, history_latents], dim=2)
                 clean_latents_4x, clean_latents_2x, clean_latents_1x = padded_history[:, :, -required_history_len:, :, :].split([16, 2, 1], dim=2)
            else:
                 # Take the last 19 frames from history
                 clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -required_history_len:, :, :].split([16, 2, 1], dim=2)

            # Always prepend the original start_latent (frame 0) to clean_latents_1x (the most recent history frame)
            clean_latents = torch.cat([start_latent_tensor.to(history_latents.device, dtype=history_latents.dtype), clean_latents_1x], dim=2)

            # vid2vid WIP (Using F1's method based on section index 'i')
            input_init_latents = None
            if initial_samples is not None:
                total_length = initial_samples.shape[2]
                # Use loop index 'i' for progress, mapping it to the vid2vid timeline
                progress = i / (total_latent_sections - 1) if total_latent_sections > 1 else 0
                start_idx = int(progress * max(0, total_length - effective_window_size))
                end_idx = min(start_idx + effective_window_size, total_length)
                # print(f"vid2vid (F1 logic) - Iteration {i}, Progress {progress:.2f}, Slice [{start_idx}:{end_idx}] of {total_length}")
                if start_idx < end_idx:
                    input_init_latents = initial_samples[:, :, start_idx:end_idx, :, :].to(device)
                else:
                     print("vid2vid - Warning: Calculated slice is empty.")

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_rel_l1_thresh)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=base_dtype, enabled=True):
                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler=sampler,
                    initial_latent=input_init_latents,
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
                    image_embeddings=current_image_embeds,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                )

            # F1 History Latents Update: Append new frames generated in this step
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
            # Increment total frame count by the number of newly generated frames
            total_generated_latent_frames += generated_latents.shape[2]

            # F1 Real History Latents Selection: Take from the end, ensuring we have `total_generated_latent_frames` count
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            if is_last_section:
                break

        transformer.to(offload_device)
        mm.soft_empty_cache()

        # Ensure final output has the expected length (or close to it)
        final_frame_count = real_history_latents.shape[2]
        expected_latent_frames = total_generated_latent_frames # F1 should generate frame by frame
        print(f"Final latent frames: {final_frame_count} (Expected based on generation: {expected_latent_frames})")

        return {"samples": real_history_latents / vae_scaling_factor},

class FramePackTimestampedTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Text prompt, use [Xs: prompt] or [Xs-Ys: prompt] for timed sections."}),
                "negative_text": ("STRING", {"multiline": False, "default": "", "dynamicPrompts": False, "tooltip": "Single negative text prompt"}),
                "total_second_length": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 1200.0, "step": 0.1, "tooltip": "Expected total video duration in seconds for timestamp calculation."}),
                "latent_window_size": ("INT", {"default": 9, "min": 1, "max": 33, "step": 1, "tooltip": "The latent window size used by the sampler for timestamp boundary snapping."}),
                "prompt_blend_sections": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1, "tooltip": "Number of latent sections (windows) over which to blend prompts when they change. 0 disables blending."}),
            },
        }
    RETURN_TYPES = ("TIMED_CONDITIONING_WITH_METADATA", "CONDITIONING",)
    RETURN_NAMES = ("positive_timed_data", "negative",)
    FUNCTION = "encode"
    CATEGORY = "FramePackWrapper/experimental"
    DESCRIPTION = """Encodes text prompts with optional timestamps for timed conditioning.

Use format: [Xs: prompt] or [Xs-Ys: prompt] where X and Y are times in seconds (e.g., 0s, 1.5s, 10s).
- [Xs: prompt]: Prompt applies from time X until the next timestamp starts (or end of video).
- [Xs-Ys: prompt]: Prompt applies specifically between time X and time Y.

Text before the first timestamp defaults to starting at 0s.
Gaps between specified timestamps are automatically filled, typically using the preceding prompt.
Timestamps are aligned to internal section boundaries based on latent_window_size.

Outputs a dictionary containing:
- timed conditioning sections: List of (start_sec, end_sec, conditioning) tuples defining the prompt for each time segment.
- total duration: The overall video length in seconds, used for time calculations.
- latent window size: The sampler's processing window size, used for aligning timestamps.
- prompt blend sections: Number of sections over which to smoothly blend between changing prompts(if you want smoother visual transitions when your timed prompts change. A higher value gives a longer, more gradual blend).
"""

    def encode(self, clip, text, negative_text, total_second_length, latent_window_size, prompt_blend_sections):
        prompt_sections = parse_timestamped_prompt_f1(text, total_second_length, latent_window_size)
        unique_prompts = sorted(list(set(section.prompt for section in prompt_sections)))
        encoded_prompts: Dict[str, List[List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]] = {}
        first_cond, first_pooled = None, None

        print(f"FramePackTimestampedTextEncode: Encoding {len(unique_prompts)} unique prompts.")
        for i, prompt in enumerate(unique_prompts):
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            if i == 0:
                 first_cond, first_pooled = cond, pooled
            encoded_prompts[prompt] = [[cond, {"pooled_output": pooled}]]

        positive_timed_list: List[Tuple[float, float, List[List[Union[torch.Tensor, Dict[str, torch.Tensor]]]]]] = []
        for section in prompt_sections:
            if section.prompt in encoded_prompts:
                encoded_cond = encoded_prompts[section.prompt]
                positive_timed_list.append((section.start_time, section.end_time, encoded_cond))
            else:
                 print(f"Warning: Prompt '{section.prompt}' not found in encoded prompts. Skipping section.")

        if not positive_timed_list:
             print("FramePackTimestampedTextEncode: Warning - No valid timed sections found. Creating a default empty section.")
             tokens = clip.tokenize("")
             cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
             if first_cond is None: first_cond, first_pooled = cond, pooled # Store shape if needed
             positive_timed_list.append((0.0, total_second_length, [[cond, {"pooled_output": pooled}]])) # Ensure list structure is maintained

        # --- Negative Conditioning ---
        if negative_text:
            tokens_neg = clip.tokenize(negative_text)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
            negative = [[cond_neg, {"pooled_output": pooled_neg}]]
        elif first_cond is not None:
            negative = [[torch.zeros_like(first_cond), {"pooled_output": torch.zeros_like(first_pooled)}]]
        else:
            print("FramePackTimestampedTextEncode: Error - Cannot create empty negative conditioning, no positive prompts found and fallback failed.")
            try:
                 tokens = clip.tokenize("")
                 cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
                 negative = [[torch.zeros_like(cond), {"pooled_output": torch.zeros_like(pooled)}]]
            except Exception as e:
                 print(f"Fallback negative shape guess failed: {e}")
                 # Minimal fallback guess
                 negative = [[torch.zeros((1, 77, 768)), {"pooled_output": torch.zeros((1, 768))}]]

        # Package results into a dictionary
        timed_data = {
            "sections": positive_timed_list,
            "total_duration": total_second_length,
            "window_size": latent_window_size,
            "blend_sections": prompt_blend_sections
        }
        return (timed_data, negative)

NODE_CLASS_MAPPINGS = {
    "FramePackSampler_F1": FramePackSampler_F1,
    "FramePackTimestampedTextEncode": FramePackTimestampedTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FramePackSampler_F1": "FramePackSampler (F1)",
    "FramePackTimestampedTextEncode": "FramePack Text Encode (Timestamped)",
}