import torch
from comfy.model_management import get_torch_device
import torch
import folder_paths # Need this for hf_dir access if kept, or remove if adapter_path is hardcoded/input
from comfy.model_management import get_torch_device
# Import helpers from the local utils.py
from .utils import convert_images_to_tensors, convert_tensors_to_images
# Import attention processor from local file
from .attention_processor import DecoupledMVRowColSelfAttnProcessor2_0


class DiffusersIGMVModelMakeup: # New Makeup Node specific to IG2MV
    def __init__(self):
        # Assuming adapter files are downloaded via huggingface cache managed elsewhere
        # self.hf_dir = folder_paths.get_folder_paths("diffusers")[0] # Potentially remove if adapter_path is fixed
        self.torch_device = get_torch_device()
        self.dtype = torch.float16

    @classmethod
    def INPUT_TYPES(s):
        # Removed adapter_name dropdown, this node is specific to ig2mv
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "scheduler": ("SCHEDULER",),
                "autoencoder": ("AUTOENCODER",),
                "load_mvadapter": ("BOOLEAN", {"default": True}),
                "adapter_path": ("STRING", {"default": "huanngzh/mv-adapter"}), # Keep path configurable? Or hardcode?
                "num_views": ("INT", {"default": 6, "min": 1, "max": 12}), # Should match pos/normal maps
            },
            "optional": {
                "enable_vae_slicing": ("BOOLEAN", {"default": True}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("PIPELINE",)
    FUNCTION = "makeup_pipeline"
    CATEGORY = "MV-Adapter/IG2MV" # Keep category consistent

    def makeup_pipeline(
        self,
        pipeline,
        scheduler,
        autoencoder,
        load_mvadapter,
        adapter_path,
        # adapter_name is removed, hardcoded below
        num_views,
        enable_vae_slicing=True,
        enable_vae_tiling=False,
    ):
        pipeline.vae = autoencoder
        pipeline.scheduler = scheduler

        if load_mvadapter:
            # Always initialize with the specific processor for ig2mv
            pipeline.init_custom_adapter(num_views=num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)

            # Hardcode the adapter weight name
            adapter_name = "mvadapter_ig2mv_sdxl.safetensors"
            # Assuming hf_dir is handled by the underlying diffusers load call if needed
            pipeline.load_custom_adapter(
                adapter_path, weight_name=adapter_name #, cache_dir=self.hf_dir # Potentially remove cache_dir
            )
            # Ensure cond_encoder is on the right device/dtype if it exists
            if hasattr(pipeline, 'cond_encoder'):
                 pipeline.cond_encoder.to(device=self.torch_device, dtype=self.dtype)

        pipeline = pipeline.to(self.torch_device, self.dtype)

        if enable_vae_slicing:
            pipeline.enable_vae_slicing()
        if enable_vae_tiling:
            pipeline.enable_vae_tiling()

        return (pipeline,)


class DiffusersIGMVSampler: # Sampler Node for Image-Guided (Position/Normal Maps)
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",), # Input pipeline from ComfyUI-MVAdapter
                "position_map": ("IMAGE",), # Input for position maps
                "normal_map": ("IMAGE",),   # Input for normal maps
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "high quality texture"},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "watermark, ugly, deformed, noisy, blurry, low contrast",
                    },
                ),
                "width": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}), # Should match map size
                "height": ("INT", {"default": 768, "min": 1, "max": 2048, "step": 1}), # Should match map size
                "steps": ("INT", {"default": 30, "min": 1, "max": 2000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                 "reference_conditioning_scale": ( # Added based on inference script
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "reference_image": ("IMAGE",),
                "lora_scale": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}), # Added LoRA scale
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sample"
    # Define a category specific to this node package
    CATEGORY = "MV-Adapter/IG2MV"

    def sample(
        self,
        pipeline,
        position_map,
        normal_map,
        prompt,
        negative_prompt,
        height,
        width,
        steps,
        cfg,
        reference_conditioning_scale,
        seed,
        reference_image=None,
        lora_scale=1.0, # Added lora_scale parameter
    ):
        # Assuming position_map and normal_map are BHWC tensors from ComfyUI
        num_views = position_map.shape[0]
        if normal_map.shape[0] != num_views:
             raise ValueError("Position map and Normal map must have the same number of views (batch size).")

        # Convert ComfyUI BHWC tensors to BCHW tensors expected by diffusers/torch
        # Normalize to [0, 1] if they aren't already (ComfyUI tensors are usually 0-1)
        pos_tensor = position_map.permute(0, 3, 1, 2).to(self.torch_device, dtype=pipeline.dtype)
        nor_tensor = normal_map.permute(0, 3, 1, 2).to(self.torch_device, dtype=pipeline.dtype)

        # Concatenate along the channel dimension (B, C, H, W) -> (B, C*2, H, W)
        # The inference script concatenated pos+normal, resulting in 6 channels
        control_images = torch.cat([pos_tensor, nor_tensor], dim=1)

        pipe_kwargs = {}
        if reference_image is not None:
            # Convert reference image tensor (BHWC) to PIL Image using local util
            ref_img_pil = convert_tensors_to_images(reference_image)[0]
            pipe_kwargs.update(
                {
                    "reference_image": ref_img_pil,
                    "reference_conditioning_scale": reference_conditioning_scale,
                }
            )

        if seed is not None and seed != -1 and isinstance(seed, int):
             pipe_kwargs["generator"] = torch.Generator(device=self.torch_device).manual_seed(seed)

        # Prepare cross-attention kwargs for LoRA scale if needed
        cross_attention_kwargs = {"num_views": num_views} # Always needed for MV-Adapter
        if lora_scale != 1.0:
            cross_attention_kwargs["scale"] = lora_scale

        # Call the pipeline (loaded and prepared by ComfyUI-MVAdapter nodes)
        images = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=num_views,
            control_image=control_images, # Pass concatenated maps here
            control_conditioning_scale=1.0, # Defaulted in inference script
            negative_prompt=negative_prompt,
            cross_attention_kwargs=cross_attention_kwargs, # Pass kwargs here
            **pipe_kwargs,
        ).images

        # Convert output PIL images back to ComfyUI BHWC tensor using local util
        return (convert_images_to_tensors(images),)

# --- Node Registration ---
NODE_CLASS_MAPPINGS = {
    "DiffusersIGMVSampler": DiffusersIGMVSampler,
    "DiffusersIGMVModelMakeup": DiffusersIGMVModelMakeup, # Added new makeup node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersIGMVSampler": "Diffusers IG MV Sampler",
    "DiffusersIGMVModelMakeup": "Diffusers IG MV Model Makeup", # Added new makeup node display name
}
