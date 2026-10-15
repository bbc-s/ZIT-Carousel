import json

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
import nodes
from comfy_extras.nodes_sd3 import EmptySD3LatentImage

from .vision import DETAIL_FIELDS, analyze_reference, local_vision_models, select_vision_model


POSES = (
    "torso leaning forward, elbows bent, both hands resting on the thighs, looking at the camera",
    "torso turned to the left, right arm reaching across the body, looking over the left shoulder",
    "both arms raised above the head, elbows bent, hands touching the hair",
    "torso leaning back, shoulders open, arms extended out to either side",
    "torso turned to the right, left hand resting on the waist, right hand lifted near the face",
    "upper body bending sideways to the left, left arm lowered, right arm curved overhead",
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
    "The same person, facial features, hair, body proportions, clothing and accessories. "
    "The same location, furniture, objects, lighting, color palette and photographic style."
)


def encode(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def shot_prompt(scene, description, variation):
    return "\n".join(part for part in (f"Photograph: {variation}.", scene.strip(), description.strip(), CONSISTENCY) if part)


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


def person_mask(image, box):
    h, w = image.shape[1:3]
    left, top, right, bottom = box
    # Leave room for arms and torso to move beyond the original silhouette.
    left, right = max(0, left - 0.12) * w, min(1, right + 0.12) * w
    top, bottom = max(0, top - 0.08) * h, min(1, bottom + 0.08) * h
    feather = max(1, min(h, w) * 0.025)
    x = torch.arange(w, device=image.device, dtype=torch.float32)
    y = torch.arange(h, device=image.device, dtype=torch.float32)
    horizontal = torch.minimum((x - left) / feather, (right - x) / feather).clamp(0, 1)
    vertical = torch.minimum((y - top) / feather, (bottom - y) / feather).clamp(0, 1)
    if left == 0:
        horizontal[x < feather] = 1
    if right == w:
        horizontal[x > w - feather] = 1
    if top == 0:
        vertical[y < feather] = 1
    if bottom == h:
        vertical[y > h - feather] = 1
    return (vertical[:, None] * horizontal[None, :]).unsqueeze(0)


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
                "denoise": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Img2img strength, not an identity lock. Around 0.3-0.45 often preserves the original pose. Try 0.6-0.75 for pose changes, with increased identity drift. Existing workflows keep their saved value."}),
                "variation_mode": (["poses", "camera", "mixed", "custom"],),
                "custom_shots": ("STRING", {"multiline": True, "default": "", "tooltip": "One instruction per variant, cycled if fewer than the batch. Used in custom mode."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "One reference image. For ZIT-Ideogram, connect the decoded sampler result, without editor overlays."}),
                "reference_description": ("STRING", {"forceInput": True, "tooltip": "Optional visual description from a local VLM, e.g. QwenVL. Text CLIP alone cannot extract identity from an image."}),
                "vision_model": (["auto", "off", *local_vision_models()], {"default": "auto", "tooltip": "Automatically inspect the first/reference image with an installed local QwenVL model. No downloads. A connected reference_description overrides automatic analysis."}),
                "reference_mode": (["auto", "whole_image", "person_inpaint"], {"default": "auto", "tooltip": "auto uses a QwenVL person box for poses and the whole image for camera/mixed. Person inpaint preserves pixels outside an expanded box; it is not silhouette segmentation."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "shots_json")
    FUNCTION = "generate"
    CATEGORY = "ZIT-Carousel"
    DESCRIPTION = "Automatically analyzes the reference with local QwenVL, reuses appearance/clothing/scene descriptions, and optionally inpaints the person region. Requires ComfyUI-QwenVL and local vision weights. Identity and camera control remain approximate."

    def generate(self, model, clip, vae, prompt, batch_size, seed, width, height,
                 steps, cfg, sampler_name, scheduler, denoise, variation_mode,
                 custom_shots, image=None, reference_description="", vision_model="auto", reference_mode="auto"):
        if not 1 <= batch_size <= 64:
            raise ValueError("ZIT Carousel: batch_size must be between 1 and 64.")
        if not 0 < denoise <= 1:
            raise ValueError("ZIT Carousel: denoise must be greater than zero and at most one.")
        shots = variations(variation_mode, custom_shots) if batch_size > 1 else ()
        if reference_mode not in ("auto", "whole_image", "person_inpaint"):
            raise ValueError("ZIT Carousel: unknown reference_mode.")
        if reference_mode == "person_inpaint" and variation_mode in ("camera", "mixed"):
            raise ValueError("ZIT Carousel: person_inpaint needs a fixed camera. Use poses/custom or whole_image for camera changes.")
        selected_model = select_vision_model(vision_model) if batch_size > 1 and vision_model != "off" and not reference_description.strip() else None
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

        analysis = None
        mask = None
        variant_scene = prompt
        if selected_model is not None:
            # This node orchestrates two independently managed model stages.
            comfy.model_management.unload_all_models()
            analysis = analyze_reference(reference, selected_model, seed)
            reference_description = "\n".join(f"{key.capitalize()}: {analysis[key]}" for key in DETAIL_FIELDS)
            variant_scene = ""  # The analyzed action replaces the reference's exact pose/framing.
            negative = encode(clip, "")
        use_inpaint = reference_mode == "person_inpaint" or (reference_mode == "auto" and variation_mode == "poses" and analysis is not None)
        if use_inpaint and batch_size > 1:
            if analysis is None:
                raise ValueError("ZIT Carousel: person_inpaint requires automatic QwenVL analysis to locate the person. Enable vision_model and disconnect reference_description.")
            mask = person_mask(reference, analysis["person_box"])

        output_device = comfy.model_management.intermediate_device()
        images = [reference.to(device=output_device)]
        for index in range(1, batch_size):
            comfy.model_management.throw_exception_if_processing_interrupted()
            variation = shots[(index - 1) % len(shots)]
            text = shot_prompt(variant_scene, reference_description, variation)
            positive = encode(clip, text)
            shot_seed = (seed + index) % (1 << 64)
            latent = {"samples": reference_latent["samples"].clone()}
            if mask is not None:
                latent["noise_mask"] = mask
            result = nodes.common_ksampler(
                model, shot_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                latent, denoise=denoise,
            )[0]
            decoded = vae.decode(result["samples"])
            if mask is not None:
                decoded = torch.lerp(reference.to(decoded), decoded, mask.movedim(0, -1).to(decoded))
            images.append(decoded.to(device=output_device, dtype=images[0].dtype))
            records.append({"index": index, "kind": "variation", "seed": shot_seed, "prompt": text})

        manifest = {"reference_description": reference_description, "analysis": analysis,
                    "vision_model": selected_model, "reference_mode": "person_inpaint" if mask is not None else "whole_image",
                    "denoise": denoise, "shots": records}
        return torch.cat(images, dim=0), json.dumps(manifest, ensure_ascii=False, indent=2)
