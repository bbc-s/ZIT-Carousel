# ZIT-Carousel

Reference-based photo variations for **Z-Image Turbo** in ComfyUI. Generate a first image or supply one. Carousel automatically inspects it with a local QwenVL model and reuses its detailed person, clothing, environment, lighting and style descriptions throughout the batch.

**Experimental:** descriptions are not identity embeddings. This does not guarantee identical faces, clothing or scene geometry across pose/viewpoint changes. Camera changes remain whole-image img2img and may retain the old viewpoint or alter other details. Visual quality has not been validated end to end.

## Installation

1. Put this directory in `ComfyUI/custom_nodes/ZIT-Carousel`.
2. Install/load **ComfyUI-QwenVL** and a complete supported vision model, such as `Qwen3-VL-2B-Instruct`, in `ComfyUI/models/LLM/Qwen-VL/<model-name>`. A custom registered `LLM` path uses its first directory, matching that extension. The single-file Qwen text encoder used by Z-Image is not a complete vision model.
3. Restart ComfyUI and add **ZIT-Carousel > ZIT Carousel**.

Carousel adds no dependencies or downloads. Automatic analysis selects supported models with local config and safetensors weights. Missing models produce an actionable error before generation. ComfyUI-QwenVL owns its dependencies and vision model lifecycle.

## Wiring

```text
Z-Image MODEL --------> model
Matching Qwen CLIP --> clip     ZIT Carousel -- images --> Save Image
Matching VAE --------> vae
Optional RGB image --> image
```

Sampling and decoding happen inside the node. Enter the prompt in Carousel. No external CLIP Text Encode, KSampler, VAE Decode or QwenVL node is needed for this path.

For ZIT-Ideogram, keep its existing graph and connect **its KSampler > VAE Decode result** to Carousel `image`, with the same MODEL/CLIP/VAE. Do not use the rectangle editor preview. Its regional masks are not transferred.

## Controls

- `batch_size`: total output count including the reference at index zero. `1` produces only the reference and skips analysis.
- `vision_model`: `auto` chooses the smallest supported local model by weight-file size. Select a local model explicitly if preferred. `off` disables analysis. No external description connection is needed.
- `reference_mode`: `auto` uses person-region inpainting for `poses` with automatic analysis, and whole-image img2img otherwise. `whole_image` always processes the full frame. `person_inpaint` requires automatic analysis and a fixed camera (`poses` or `custom`).
- `denoise`: default `0.6`. At `0.3-0.45` the pose can remain almost exact. Try `0.6`, then `0.7-0.75` if needed; more noise can also change identity and clothing. These are troubleshooting values, not visually measured guarantees. Existing workflow values must be changed manually.
- `variation_mode`: `poses` varies torso/arms, `camera` varies viewpoint/framing, `mixed` combines both, and `custom` takes one target pose/framing instruction per line. Presets cycle after six variants with different seeds. They are fixed presets, not model-planned shots.
- `reference_description`: optional external text that overrides automatic analysis when nonempty. This path cannot use automatic person inpainting because it has no person box.

Start with existing Turbo sampler settings: 9 steps, CFG 1, Euler/simple. Width/height apply only when generating a reference. A supplied reference must be one RGB image; dimensions are resized down to multiples of eight if needed.

## Analysis and inpainting

QwenVL runs once per batch and returns structured descriptions plus an estimated whole-person bounding box. Exact pose, gaze, expression and camera angle are excluded from the requested descriptions. The original prompt generates the first image. Subsequent prompts use analyzed descriptions and coarse activity instead, avoiding repeated original limb positions/framing. Invalid analysis fails clearly instead of silently falling back to text-only variations.

For person inpainting, Carousel expands/feathers the estimated box, supplies it as the latent noise mask, and composites the result over the reference. **Pixels outside the expanded mask remain exactly the reference.** This is a rectangular region, not silhouette segmentation: background inside it can change, and new limbs outside it can be clipped. Check the estimated box in `shots_json`. This path is intended for one visible person.

Camera/mixed modes use the whole frame because another viewpoint needs a different background projection. They preserve scene descriptions, not exact background pixels. Text analysis and inpainting do not provide geometric camera control or an independent face lock.

Every variation starts from the original reference latent. Sampling is sequential. ComfyUI models are offloaded before analysis; QwenVL is called with `keep_model_loaded=false` to release its model before sampling. Final image-batch memory still grows with batch size.

`shots_json` records analysis, selected vision model, person box, actual reference mode, seeds and final prompts. `images` connects directly to Preview Image or Save Image.

## Examples and checks

Drag `examples/text_to_carousel.json` into ComfyUI for the automatic single-node path. `examples/visual_context.json` retains an optional external QwenVL description path without automatic inpainting. Files ending in `_api.json` are equivalent prompts for the local `/prompt` endpoint. Adjust model filenames to installed files.

Run `python tests/test_carousel.py` in ComfyUI's Python environment. Tests cover reference reuse, batch counts/seeds, analysis orchestration, mask compositing, local model selection, invalid VLM output and cancellation. They do not measure identity, pose or camera quality.
