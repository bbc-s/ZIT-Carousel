# ZIT-Carousel

One ComfyUI node for reference-based photo series. Z-Image Turbo generates the first image, or you supply an RGB image. Subsequent images use **pose ControlNet** or **Qwen Image Edit**, depending on the mode. The old text-only img2img variation path has been replaced.

**Experimental integration, not a verified identity lock.** Wiring and data-flow tests pass; generated-image quality has not been tested for this version. Inpainted clothing can change. Another camera view requires synthesizing unseen surfaces and cannot preserve exact background pixels.

## Wiring

```text
Z-Image MODEL ------> model
Z-Image CLIP -------> clip     ZIT Carousel -- images --> Save Image
Z-Image VAE --------> vae
Optional RGB image -> image
```

Sampling, conditioning and decoding happen inside Carousel. No additional ControlNet or editor nodes are needed. For ZIT-Ideogram, connect its **decoded sampler image**, not the rectangle editor preview. `batch_size` includes the reference as image zero. Each variation uses that original reference, never the previous result.

## Modes

| Mode | Automatic backend | Behavior |
| --- | --- | --- |
| `poses` | Z-Image Turbo + Union ControlNet | Six target arm-pose templates, fitted to estimated person/head boxes and seated/standing posture. Fixed camera/head. |
| `camera` | Qwen Image Edit 2511 | Actual reference image conditioning; asks to change viewpoint/framing while retaining the pose and clothes. |
| `mixed` | Qwen Image Edit 2511 | Reference-conditioned pose and viewpoint edits. |
| `custom` | Qwen Image Edit 2511 | One edit instruction per line. |

Set `variation_backend=reference_edit` to use the editor for freer pose changes too. Presets cycle after six variants. Pose guides are generated templates, **not detected skeletons or a 3D reconstruction**. The ControlNet path supports one seated/standing person with enough visible body; use reference editing for other postures. Close crops, occlusion and inaccurate vision boxes can give poor guides.

For ControlNet, `reference_mode=auto` masks the union of the source person box and target skeleton, protects the original head rectangle, then composites the result over the reference. Pixels outside the mask remain unchanged. Background inside the mask can change; head protection also prevents changing head position/expression and can leave seams. `whole_image` disables this protection. `person_inpaint` can explicitly constrain reference editing with a fixed camera, but cannot be used with camera/mixed.

## Resolution and settings

- **Width/height now apply to supplied references too.** The image is resized and center-cropped to fit, preserving aspect ratio. Cropping can remove content at the edges.
- `use_reference_resolution=true` instead uses the reference dimensions, rounded down to multiples of eight. With no reference, width/height always apply.
- Set **`denoise=1.0`** for ControlNet pose changes. Existing workflows retain saved values such as 0.4, which still constrain movement. The new default is 1.0.
- ZIT reference/pose defaults: 9 steps, CFG 1, Euler/simple; `control_strength=0.8`.
- Editor defaults: `edit_steps=20`, `edit_cfg=4`, with the selected sampler/scheduler and full denoise. ZIT's denoise does not restrict the editor.
- `vision_model=auto` runs a local QwenVL analysis once. Pose control requires it for geometry. In editor mode, a supplied `reference_description` skips this separate analysis; `off` is also valid because the editor itself consumes the reference pixels.
- `reference_description` supplements automatic pose analysis when connected. It cannot replace geometry.
- `batch_size=1` skips variant model loading and analysis.

## Local models

Install in `ComfyUI/custom_nodes/ZIT-Carousel`, then restart ComfyUI. Current native `ZImageFunControlnet` and `TextEncodeQwenImageEditPlus` support is required. No additional Python dependencies or automatic downloads are added.

For automatic analysis, install **ComfyUI-QwenVL** and a complete supported model such as `Qwen3-VL-2B-Instruct` under `models/LLM/Qwen-VL/`. This is separate from the ZIT text encoder.

The `auto` model selections resolve these exact files; advanced selectors allow explicitly choosing compatible local alternatives:

| Purpose | File location | Download |
| --- | --- | --- |
| Pose ControlNet | `models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors` | [Alibaba-PAI Union 2.1](https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1/tree/main), about 2.02 GB |
| Reference editor | `models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` | [Comfy-Org Qwen Image Edit](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/tree/main/split_files/diffusion_models), about 20.5 GB |
| Editor encoder | `models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | [Comfy-Org encoder](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main/split_files/text_encoders), about 9.38 GB |
| Editor VAE | `models/vae/qwen_image_vae.safetensors` | [Comfy-Org Qwen Image VAE](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/tree/main/split_files/vae) |

Pose-only use does not require editor weights. Missing files stop the node before first-image generation; there is no silent fallback to the old behavior. Loaders use ComfyUI's model management and filename resolvers. The editor is substantially heavier than ZIT, especially with CPU offloading on an 8 GB GPU; runtime/memory have not been benchmarked. No models or image tensors are cached by Carousel across executions.

## Examples and verification

`examples/text_to_carousel.json` is the automatic pose workflow. `examples/visual_context.json` demonstrates an external description with reference editing. Matching `_api.json` files are API prompts. Update ZIT model filenames for your installation.

`shots_json` records backend, model filenames, analysis/boxes, output dimensions, seeds, prompts and effective sampling settings. `images` is the complete decoded batch.

Run `python tests/test_carousel.py` using ComfyUI's Python environment. Tests exercise dimensions, reference reuse, masks, distinct pose guides, backend routing, local file validation, cancellation and native adapter contracts with mocked sampling/model loading. They do not generate pictures or prove visual consistency.
