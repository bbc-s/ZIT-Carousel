# ZIT-Carousel

Local reference-based photo variations for **Z-Image Turbo** in ComfyUI. Generate a first image or supply one, then produce a batch of pose/framing variations from that same reference. No new dependencies, downloads, or network requests are added by this extension.

**Experimental:** this is img2img, not a trained identity adapter or face recognition system. Turbo cannot guarantee the same face, body, clothing, bench, or lighting across large pose/viewpoint changes. Low denoise preserves more of the reference but limits motion; high denoise allows more change and can replace the person. Prompt instructions and a visual description improve guidance but do not lock identity.

## Installation

Place this directory at `ComfyUI/custom_nodes/ZIT-Carousel`, restart ComfyUI, and add **ZIT-Carousel → ZIT Carousel**. Use your existing Z-Image Turbo MODEL, matching Qwen CLIP, and VAE loaders. The node runs sampling internally.

## Basic use

1. Connect `model`, `clip`, and `vae`. Enter a detailed scene prompt, including the person's appearance, clothing, location, light, and style.
2. Set `batch_size` to the total number of output images. Image zero is always the reference; the remaining images are variations. `batch_size=1` generates just the reference.
3. Optionally connect a single RGB `image`. Its dimensions take precedence over width/height. Dimensions not divisible by eight are resized down to the nearest multiple of eight before encoding.
4. Connect `images` to Save Image or Preview Image. Optional `shots_json` records seeds and the exact variant prompts.

Use 9 steps, CFG 1 and Euler/simple as a starting point. The default denoise is now 0.6. At 0.3?0.45 img2img can retain the original pose almost exactly; try 0.6, then 0.7?0.75 if the body still does not move. These are troubleshooting values, not visually validated guarantees. More noise can also change the face, clothes and background. Restart ComfyUI after updating; existing nodes and saved workflows retain their previous denoise value, so change it manually.

`poses` now requests distinct torso and arm positions rather than small gaze/expression changes. `camera` requests framing/viewpoint changes; `mixed` combines both. The requested pose appears first in each variant prompt. Keep the main prompt focused on the person, action, setting and style; exact limb positions in that prompt or in a VLM description can conflict with the variation. Automatic instructions are fixed shot presets, not semantic reasoning. `custom` accepts one complete target pose/framing description per line, cycled as needed. Presets also cycle after six variations with different seeds.

This change improves pose instructions, but does not add independent identity and pose control. If stronger denoise changes the person before it changes the pose, this plain img2img approach cannot meet that case; a compatible pose-control and identity-conditioning model path is needed.

Every variation starts from the original reference latent, never from the preceding variant. Sampling is sequential to avoid holding a whole latent batch on the GPU. ComfyUI owns loading/offloading and sampling; the final image batch still consumes memory proportional to batch size.

## Automatic visual context with QwenVL

Text CLIP encodes text; it does not inspect reference images. To extract visible characteristics automatically, use the separately installed **ComfyUI-QwenVL** node in this two-stage graph:

```text
ZIT Carousel A (batch_size=1) ── images ──┬── QwenVL ── RESPONSE ──┐
                                      │                        │ reference_description
                                      └── image ── ZIT Carousel B ── Save Image
```

Both Carousel nodes share MODEL/CLIP/VAE and the scene prompt. Carousel B has the desired total batch size. For an existing image, replace Carousel A with Load Image. Use a locally available QwenVL model and set `keep_model_loaded=false` to let that extension release its model before image generation. QwenVL is optional and independently managed; its own loader may download missing models, so configure a model already present locally if offline operation is required.

Suggested QwenVL custom prompt:

> Describe only visible, stable visual details for reproducing this photograph: facial appearance, hair, body proportions, clothing and accessories, location, furniture and objects, their colors and materials, light direction and softness, palette and photographic style. Do not guess hidden details or identify the person. Write concise English prose. Omit the exact pose, gaze, expression and camera angle because these will vary.

The resulting description stays fixed throughout Carousel B's batch. This graph automatically analyzes the first image once; the Carousel node itself does not contain a vision model. VLM descriptions can be wrong and are not biometric embeddings.

## ZIT-Ideogram

Keep its regional workflow intact: **ZIT-Ideogram → KSampler → VAE Decode → Carousel `image`**. Use the decoded generated image, not the rectangle editor preview. Connect the same MODEL/CLIP/VAE to Carousel. Optionally send that decoded image through QwenVL into `reference_description` too. Region masks are not transferred: Carousel varies the whole photograph.

Drag `examples/text_to_carousel.json` into the ComfyUI canvas for basic generation, or `examples/visual_context.json` for the QwenVL stage. Adjust model names to installed files. The files ending in `_api.json` contain the equivalent prompts for the local `/prompt` endpoint.

## Validation

Run `python tests/test_carousel.py` using ComfyUI's Python environment. Tests cover reference reuse, seed rollover, batch counts, image resizing, invalid input, prompt context, and cancellation. They do not measure identity consistency or visual quality.
