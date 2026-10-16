# ZIT-Carousel

A single ComfyUI node using Z-Image Turbo, local reference analysis and native masked inpainting. Generate a first image or supply one, then produce a batch from the same reference.

**No ControlNet, second diffusion model or additional model downloads.** Automatic analysis uses the existing ComfyUI-QwenVL extension and a locally installed vision model. CLIP text encoding alone cannot inspect an image.

## Wiring

```text
Z-Image MODEL ------> model
Z-Image CLIP -------> clip     ZIT Carousel -- images --> Save Image
Z-Image VAE --------> vae
Optional RGB image -> image
```

Sampling and decoding are internal. For ZIT-Ideogram, supply the decoded sampler image, not the rectangle editor preview. `batch_size` includes the reference at index zero. Every variant starts from that reference, never the previous result.

## Automatic inpainting

With `variation_mode=poses`, `reference_mode=auto` and `vision_model=auto`:

1. The local vision model describes appearance, clothing, environment, lighting and style, and estimates person/head rectangles.
2. Carousel expands the person rectangle to leave room for movement and excludes the head.
3. Native `VAEEncodeForInpaint` clears the editable pixels before VAE encoding and creates the sampler noise mask. This removes the old pose from that region instead of retaining its original latent as ordinary img2img does.
4. ZIT redraws the region using the description and target pose instruction.
5. Compositing restores the protected head and pixels outside the mask from the reference.

The mask is an estimated rectangle, **not automatic silhouette segmentation**. Background inside it can change and limbs outside it can be clipped. Head protection fixes its position and expression and may leave seams. Clothing inside the editable region must be regenerated from the description; identical garment details and correct target poses are not guaranteed. Image quality has not been tested for this version.

## Settings

- `width` / `height` apply to supplied references too. Resizing preserves aspect ratio with a center crop, which may remove edge content.
- `use_reference_resolution=true` uses reference dimensions rounded down to multiples of eight instead. Without an input image, the configured dimensions always apply.
- `denoise=1.0` is the default for redrawing the cleared inpaint region. Saved workflows retain their old values. This is not an identity-strength setting.
- `poses` cycles six arm-pose instructions with a fixed camera/head. They are text instructions, not enforced skeletons.
- `camera` / `mixed` retain experimental whole-image img2img. **The earlier camera-consistency problem is not solved by native inpainting.** These modes can retain the old viewpoint or change identity/background. A different viewpoint cannot preserve exact background pixels.
- `custom` cycles one instruction per line. Explicit `person_inpaint` is available for fixed-camera edits; it does not protect the head automatically in custom mode.
- `reference_mode=whole_image` disables masking and uses ordinary img2img. `person_inpaint` is incompatible with camera/mixed.
- `vision_model=auto` selects a supported local vision model. Inpainting requires analysis even with a supplied `reference_description`; that description supplements the analysis. Whole-image mode can use a description or `vision_model=off` without automatic analysis.
- `batch_size=1` skips analysis and variation sampling.

Start with ZIT's 9 steps, CFG 1, Euler/simple. For the automatic inpaint path, use `poses`, `auto`, `vision_model=auto`, and `denoise=1.0`.

## Installation and checks

Place this folder under `ComfyUI/custom_nodes/` and restart ComfyUI. For automatic analysis, ComfyUI-QwenVL and a complete supported local model such as `Qwen3-VL-2B-Instruct` in `models/LLM/Qwen-VL/` must already be installed. Carousel does not download weights or add Python dependencies.

The ControlNet/editor integration and its model selectors have been removed. After updating a workflow saved with that short-lived version, recreate the Carousel node if stale widgets remain. MODEL/CLIP/VAE inputs and IMAGE/STRING outputs are unchanged.

`examples/text_to_carousel.json` demonstrates automatic inpainting; `examples/visual_context.json` demonstrates external descriptions with whole-image img2img. Matching `_api.json` files are API prompts. Adjust model filenames to your installation.

`shots_json` records analysis, actual processing mode, dimensions, prompts, seeds and sampling settings. Run `python tests/test_carousel.py` with ComfyUI's Python. Tests use native inpaint preprocessing and mocked sampling to check region clearing, protected pixels, reference reuse, dimensions and analysis validation. They do not generate images or measure visual consistency.
