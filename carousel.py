import json

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
import nodes
from comfy_extras.nodes_sd3 import EmptySD3LatentImage


POSES = (
    "a slight turn of the head, eyes looking toward the camera",
    "shoulders turned slightly, a relaxed natural expression",
    "a different natural hand position, gaze slightly away from the camera",
    "head tilted slightly, a subtle smile",
    "upper body turned slightly toward the camera, relaxed hands",
    "a candid expression, gaze directed to the side",
)
ANGLES = (
    "camera at eye level, a slight three-quarter view from the left",
    "camera at eye level, a slight three-quarter view from the right",
    "camera slightly above eye level",
    "camera slightly below eye level",
    "a closer framing from the same camera direction",
    "a wider framing from the same camera direction",
)
CONSISTENCY = (
    "One photograph from the same photo session. Preserve the person's facial features, "
    "hair, body proportions, clothing and accessories, the same location, furniture, "
    "objects, lighting, color palette and photographic style. Keep the action described "
    "in the scene. Change only the requested pose or camera framing."
)


def encode(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def shot_prompt(scene, description, variation):
    return "\n".join(part for part in (scene.strip(), description.strip(), CONSISTENCY, variation) if part)


def variations(mode, custom_shots):
    if mode == "custom":
        shots = [line.strip() for line in custom_shots.splitlines() if line.strip()]
        if not shots:
            raise ValueError("ZIT Carousel: custom mode requires one shot instruction per line.")
        return shots
    if mode == "poses":
        return POSES
    if mode == "camera":
        return ANGLES
    if mode == "mixed":
        return tuple(f"{pose}; {angle}" for pose, angle in zip(POSES, ANGLES))
    raise ValueError(f"ZIT Carousel: unknown variation mode {mode!r}.")


class ZITCarousel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"multiline": True, "default": "A woman sitting on a wooden bench in a park, soft afternoon light, natural photography."}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip": "Total images, including the reference as the first image."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "Used only when no reference image is connected."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 0.4, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Variation strength. Lower preserves appearance; higher allows larger changes and more identity drift."}),
                "variation_mode": (["poses", "camera", "mixed", "custom"],),
                "custom_shots": ("STRING", {"multiline": True, "default": "", "tooltip": "One instruction per variant, cycled if fewer than the batch. Used in custom mode."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "One reference image. For ZIT-Ideogram, connect the decoded sampler result, without editor overlays."}),
                "reference_description": ("STRING", {"forceInput": True, "tooltip": "Optional visual description from a local VLM, e.g. QwenVL. Text CLIP alone cannot extract identity from an image."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "shots_json")
    FUNCTION = "generate"
    CATEGORY = "ZIT-Carousel"
    DESCRIPTION = "Reference-based Z-Image Turbo variations. Identity preservation is approximate, not a face lock. All variants start from the same reference."

    def generate(self, model, clip, vae, prompt, batch_size, seed, width, height,
                 steps, cfg, sampler_name, scheduler, denoise, variation_mode,
                 custom_shots, image=None, reference_description=""):
        if not 1 <= batch_size <= 64:
            raise ValueError("ZIT Carousel: batch_size must be between 1 and 64.")
        if not 0 < denoise <= 1:
            raise ValueError("ZIT Carousel: denoise must be greater than zero and at most one.")
        shots = variations(variation_mode, custom_shots) if batch_size > 1 else ()
        negative = encode(clip, "")
        records = []
        if image is None:
            positive = encode(clip, "\n".join(part for part in (prompt.strip(), reference_description.strip()) if part))
            latent = EmptySD3LatentImage.execute(width, height, 1)[0]
            reference_latent = nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
            )[0]
            reference = vae.decode(reference_latent["samples"])
            records.append({"index": 0, "kind": "generated_reference", "seed": seed})
        else:
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
                raise ValueError("ZIT Carousel: provide one RGB reference image, not an image batch.")
            h, w = image.shape[1:3]
            if h < 8 or w < 8:
                raise ValueError("ZIT Carousel: reference dimensions must be at least 8 pixels.")
            reference = image
            if h % 8 or w % 8:
                reference = comfy.utils.common_upscale(image.movedim(-1, 1), w // 8 * 8, h // 8 * 8, "area", "disabled").movedim(1, -1)
            reference_latent = {"samples": vae.encode(reference)} if batch_size > 1 else None
            records.append({"index": 0, "kind": "input_reference", "seed": None})

        output_device = comfy.model_management.intermediate_device()
        images = [reference.to(device=output_device)]
        for index in range(1, batch_size):
            comfy.model_management.throw_exception_if_processing_interrupted()
            variation = shots[(index - 1) % len(shots)]
            text = shot_prompt(prompt, reference_description, variation)
            positive = encode(clip, text)
            shot_seed = (seed + index) % (1 << 64)
            result = nodes.common_ksampler(
                model, shot_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                {"samples": reference_latent["samples"].clone()}, denoise=denoise,
            )[0]
            decoded = vae.decode(result["samples"])
            images.append(decoded.to(device=output_device, dtype=images[0].dtype))
            records.append({"index": index, "kind": "variation", "seed": shot_seed, "prompt": text})

        manifest = {"reference_description": reference_description, "denoise": denoise, "shots": records}
        return torch.cat(images, dim=0), json.dumps(manifest, ensure_ascii=False, indent=2)
