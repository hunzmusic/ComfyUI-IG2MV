![image info](../../images/mv.png)

# ComfyUI-IG2MV

This custom node package provides nodes specifically for using the `mvadapter_ig2mv_sdxl.safetensors` adapter within ComfyUI. This adapter is designed for image-guided multi-view generation, typically used for creating textures from 3D mesh renders (position and normal maps).

## Nodes

*   **Diffusers IG MV Model Makeup:** Prepares the diffusion pipeline by loading the specific `mvadapter_ig2mv_sdxl.safetensors` adapter and initializing it with the required attention processor. This node will automatically download the adapter file from the specified Hugging Face repository path (e.g., `huanngzh/mv-adapter`) if it's not already present in your diffusers cache.
*   **Diffusers IG MV Sampler:** Performs the sampling process using the prepared pipeline, taking position maps and normal maps as control inputs.

## Dependencies

This node package **requires** the following other custom nodes to be installed and working correctly:

1.  **ComfyUI-MVAdapter:** Provides the core pipeline loading nodes (`Diffusers Pipeline Loader`, `Diffusers Vae Loader`, `Diffusers Scheduler Loader`, `Lora Model Loader` etc.) and the underlying MV-Adapter pipeline implementation that this node relies on. (Please ensure you have the main ComfyUI-MVAdapter installed).
2.  **ComfyUI-Hunyuan3DWrapper:** Required for generating the **Position Map** and **Normal Map** image batches that serve as input to the `Diffusers IG MV Sampler` node. (Please ensure you have this installed).

*(Note: You will need to find and install these dependencies separately if you haven't already. The necessary Python packages like `diffusers` and `einops` should be covered by installing the requirements for `ComfyUI-MVAdapter`.)*

## Important Limitation: 6 Views Only

The underlying `mvadapter_ig2mv_sdxl.safetensors` adapter and its specific attention mechanism (`DecoupledMVRowColSelfAttnProcessor2_0`) are **hardcoded to work with exactly 6 views**.

Therefore, you **must** provide batches containing exactly 6 images for both the `position_map` and `normal_map` inputs to the `Diffusers IG MV Sampler` node. Providing a different number of views will result in errors or incorrect output.

## Example Workflow

1.  Load base model, VAE, scheduler using nodes from `ComfyUI-MVAdapter`.
2.  Optionally load LoRAs using `LoraModelLoader` from `ComfyUI-MVAdapter`.
3.  Use nodes from `ComfyUI-Hunyuan3DWrapper` to generate a batch of 6 position maps and a batch of 6 normal maps for your desired views.
4.  Feed the pipeline, scheduler, and VAE into the **`Diffusers IG MV Model Makeup`** node (from this package).
5.  Feed the output pipeline from the makeup node into the **`Diffusers IG MV Sampler`** node (from this package).
6.  Connect the position map batch to the `position_map` input of the sampler.
7.  Connect the normal map batch to the `normal_map` input of the sampler.
8.  Provide prompt, negative prompt, reference image (optional), LoRA scale (optional) etc. to the sampler.
9.  Connect the output images to a preview node.
