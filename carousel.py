import json

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
import nodes
from comfy_extras.nodes_sd3 import EmptySD3LatentImage

from .vision import DETAIL_FIELDS, analyze_reference, local_vision_models, select_vision_model
POSES = (
    "both hands resting on the thighs, elbows slightly bent",
    "both hands on the hips, elbows pointing outward",
    "both arms raised overhead with bent elbows",
    "both arms stretched out to the sides",
    "left hand raised beside the head, right hand on the hip",
    "left arm raised overhead, right arm hanging down",
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


def protect_head(mask, box):
    h, w = mask.shape[-2:]
    left, top, right, bottom = box
    padding = .02
    mask[:, max(0, int((top - padding) * h)):min(h, int((bottom + padding) * h) + 1),
         max(0, int((left - padding) * w)):min(w, int((right + padding) * w) + 1)] = 0
    return mask


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
                "width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "Output width, including supplied references unless use_reference_resolution is enabled. Reference is center-cropped to fit."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Inpaint strength. The editable region is cleared before encoding; 1.0 redraws it fully. Whole-image mode uses ordinary img2img and can retain the original pose."}),
                "variation_mode": (["poses", "camera", "mixed", "custom"],),
                "custom_shots": ("STRING", {"multiline": True, "default": "", "tooltip": "One instruction per variant, cycled if fewer than the batch. Used in custom mode."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "One reference image. For ZIT-Ideogram, connect the decoded sampler result, without editor overlays."}),
                "reference_description": ("STRING", {"forceInput": True, "tooltip": "Optional visual description from a local VLM, e.g. QwenVL. Text CLIP alone cannot extract identity from an image."}),
                "vision_model": (["auto", "off", *local_vision_models()], {"default": "auto", "tooltip": "Use an installed local QwenVL model to describe the reference and locate the person/head. No new model downloads."}),
                "reference_mode": (["auto", "whole_image", "person_inpaint"], {"default": "auto", "tooltip": "auto inpaints the person for poses, preserving the head and outside pixels. Camera/mixed use approximate whole-image img2img. Person inpaint needs vision analysis."}),
                "use_reference_resolution": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "shots_json")
    FUNCTION = "generate"
    CATEGORY = "ZIT-Carousel"
    DESCRIPTION = "ZIT with automatic local reference analysis and native masked inpainting. Preserves pixels outside the mask and optionally the head. Pose, clothing and camera consistency remain approximate; no ControlNet or second generator required."

    def generate(self, model, clip, vae, prompt, batch_size, seed, width, height,
                 steps, cfg, sampler_name, scheduler, denoise, variation_mode,
                 custom_shots, image=None, reference_description="", vision_model="auto", reference_mode="auto",
                 use_reference_resolution=False):
        if not 1 <= batch_size <= 64:
            raise ValueError("ZIT Carousel: batch_size must be between 1 and 64.")
        if not 0 < denoise <= 1:
            raise ValueError("ZIT Carousel: denoise must be greater than zero and at most one.")
        if not (8 <= width <= nodes.MAX_RESOLUTION and 8 <= height <= nodes.MAX_RESOLUTION and width % 8 == 0 and height % 8 == 0):
            raise ValueError("ZIT Carousel: output dimensions must be multiples of eight.")
        shots = variations(variation_mode, custom_shots) if batch_size > 1 else ()
        if reference_mode not in ("auto", "whole_image", "person_inpaint"):
            raise ValueError("ZIT Carousel: unknown reference_mode.")
        if reference_mode == "person_inpaint" and variation_mode in ("camera", "mixed"):
            raise ValueError("ZIT Carousel: person_inpaint needs a fixed camera. Use poses/custom or whole_image for camera changes.")
        use_inpaint = reference_mode == "person_inpaint" or (reference_mode == "auto" and variation_mode == "poses")
        if batch_size > 1 and use_inpaint and vision_model == "off":
            raise ValueError("ZIT Carousel: enable vision_model for automatic person/head masks, or select whole_image.")
        selected_model = select_vision_model(vision_model) if batch_size > 1 and vision_model != "off" and (use_inpaint or not reference_description.strip()) else None
        records = []
        if image is None:
            negative = encode(clip, "")
            positive = encode(clip, "\n".join(part for part in (prompt.strip(), reference_description.strip()) if part))
            latent = EmptySD3LatentImage.execute(width, height, 1)[0]
            reference_latent = nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent,
            )[0]
            reference = vae.decode(reference_latent["samples"])
            del reference_latent, latent
            records.append({"index": 0, "kind": "generated_reference", "seed": seed})
        else:
            if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] != 3:
                raise ValueError("ZIT Carousel: provide one RGB reference image, not an image batch.")
            h, w = image.shape[1:3]
            if h < 8 or w < 8:
                raise ValueError("ZIT Carousel: reference dimensions must be at least 8 pixels.")
            target_w, target_h = (w // 8 * 8, h // 8 * 8) if use_reference_resolution else (width, height)
            reference = image
            if (w, h) != (target_w, target_h):
                reference = comfy.utils.common_upscale(image.movedim(-1, 1), target_w, target_h, "bicubic", "center").movedim(1, -1).clamp(0, 1)
            records.append({"index": 0, "kind": "input_reference", "seed": None})

        analysis = None
        mask = None
        variant_scene = prompt
        if selected_model is not None:
            # This node orchestrates two independently managed model stages.
            comfy.model_management.unload_all_models()
            analysis = analyze_reference(reference, selected_model, seed)
            reference_description = "\n".join(f"{key.capitalize()}: {analysis[key]}" for key in DETAIL_FIELDS) + "\n" + reference_description.strip()
            variant_scene = ""  # The analyzed action replaces the reference's exact pose/framing.
        if use_inpaint and batch_size > 1:
            if analysis is None:
                raise ValueError("ZIT Carousel: person_inpaint requires automatic QwenVL analysis. Enable vision_model.")
            mask = person_mask(reference, analysis["person_box"])
            if variation_mode == "poses":
                if "head_box" not in analysis:
                    raise ValueError("ZIT Carousel: automatic pose inpainting requires head_box from QwenVL.")
                mask = protect_head(mask, analysis["head_box"])

        if batch_size > 1:
            negative = encode(clip, "")
            if mask is not None:
                reference_latent = nodes.VAEEncodeForInpaint().encode(vae, reference, mask, grow_mask_by=0)[0]
            else:
                reference_latent = {"samples": vae.encode(reference)}

        output_device = comfy.model_management.intermediate_device()
        images = [reference.to(device=output_device)]
        for index in range(1, batch_size):
            comfy.model_management.throw_exception_if_processing_interrupted()
            variation = shots[(index - 1) % len(shots)]
            text = shot_prompt(variant_scene, reference_description, variation)
            shot_seed = (seed + index) % (1 << 64)
            if use_inpaint and variation_mode == "poses":
                text += "\nKeep the original camera, head position, gaze and expression."
            elif variation_mode == "camera":
                text += "\nKeep the person's pose and clothing; change only the camera viewpoint."
            positive = encode(clip, text)
            latent = {key: value.clone() for key, value in reference_latent.items()}
            result = nodes.common_ksampler(
                model, shot_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                latent, denoise=denoise,
            )[0]
            decoded = vae.decode(result["samples"])
            if mask is not None:
                decoded = torch.lerp(reference.to(decoded), decoded, mask.movedim(0, -1).to(decoded))
            images.append(decoded.to(device=output_device, dtype=images[0].dtype))
            records.append({"index": index, "kind": "variation", "seed": shot_seed, "prompt": text,
                            "denoise": denoise, "steps": steps, "cfg": cfg})
            del result, positive, latent, decoded

        manifest = {"reference_description": reference_description, "analysis": analysis,
                    "vision_model": selected_model, "reference_mode": "person_inpaint" if mask is not None else "whole_image",
                    "backend": "native_inpaint" if mask is not None else "img2img",
                    "width": reference.shape[2], "height": reference.shape[1],
                    "use_reference_resolution": use_reference_resolution, "shots": records}
        return torch.cat(images, dim=0), json.dumps(manifest, ensure_ascii=False, indent=2)
