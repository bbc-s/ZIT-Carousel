import json

import torch

import comfy.model_management
import comfy.samplers
import comfy.utils
import nodes
from comfy_extras.nodes_sd3 import EmptySD3LatentImage

from .vision import DETAIL_FIELDS, analyze_reference, local_vision_models, select_vision_model
from .backends import model_choices, resolve_models, load_editor, edit_conditioning, load_controlnet, apply_pose
from .poses import POSES, pose_guide, protect_head


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
                "width": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "Output width, including supplied references unless use_reference_resolution is enabled. Reference is center-cropped to fit."}),
                "height": ("INT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "steps": ("INT", {"default": 9, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "ControlNet inpaint strength. Use 1.0 to release the original pose. Saved low values still restrict movement. Reference editing always uses a full noise schedule."}),
                "variation_mode": (["poses", "camera", "mixed", "custom"],),
                "custom_shots": ("STRING", {"multiline": True, "default": "", "tooltip": "One instruction per variant, cycled if fewer than the batch. Used in custom mode."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "One reference image. For ZIT-Ideogram, connect the decoded sampler result, without editor overlays."}),
                "reference_description": ("STRING", {"forceInput": True, "tooltip": "Optional visual description from a local VLM, e.g. QwenVL. Text CLIP alone cannot extract identity from an image."}),
                "vision_model": (["auto", "off", *local_vision_models()], {"default": "auto", "tooltip": "Local reference analysis. Required for ControlNet pose geometry even with a connected description."}),
                "reference_mode": (["auto", "whole_image", "person_inpaint"], {"default": "auto", "tooltip": "auto protects the head and outside-person region for ControlNet poses. Reference editing uses the whole frame. Explicit person_inpaint requires analysis and a fixed camera."}),
                "use_reference_resolution": ("BOOLEAN", {"default": False}),
                "variation_backend": (["auto", "reference_edit"], {"default": "auto", "tooltip": "auto: ZIT ControlNet for poses; Qwen Image Edit for camera/mixed/custom. reference_edit also allows freer poses."}),
                "pose_controlnet": (model_choices("pose_controlnet"),),
                "control_strength": ("FLOAT", {"default": 0.8, "min": 0.01, "max": 2.0, "step": 0.05}),
                "edit_model": (model_choices("edit_model"),),
                "edit_clip": (model_choices("edit_clip"),),
                "edit_vae": (model_choices("edit_vae"),),
                "edit_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "edit_cfg": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "shots_json")
    FUNCTION = "generate"
    CATEGORY = "ZIT-Carousel"
    DESCRIPTION = "ZIT generates the reference. Variants use pose ControlNet with head/background protection, or Qwen Image Edit with actual image conditioning. Local weights required; no downloads. Experimental: identity, clothing and camera changes need visual verification."

    def generate(self, model, clip, vae, prompt, batch_size, seed, width, height,
                 steps, cfg, sampler_name, scheduler, denoise, variation_mode,
                 custom_shots, image=None, reference_description="", vision_model="auto", reference_mode="auto",
                 use_reference_resolution=False, variation_backend="auto", pose_controlnet="auto", control_strength=0.8,
                 edit_model="auto", edit_clip="auto", edit_vae="auto", edit_steps=20, edit_cfg=4.0):
        if not 1 <= batch_size <= 64:
            raise ValueError("ZIT Carousel: batch_size must be between 1 and 64.")
        if not 0 < denoise <= 1:
            raise ValueError("ZIT Carousel: denoise must be greater than zero and at most one.")
        if variation_backend not in ("auto", "reference_edit"):
            raise ValueError("ZIT Carousel: unknown variation_backend.")
        if not (8 <= width <= nodes.MAX_RESOLUTION and 8 <= height <= nodes.MAX_RESOLUTION and width % 8 == 0 and height % 8 == 0):
            raise ValueError("ZIT Carousel: output dimensions must be multiples of eight.")
        if not 0 < control_strength <= 2 or not 1 <= edit_steps <= 100 or not 1 <= edit_cfg <= 20:
            raise ValueError("ZIT Carousel: invalid ControlNet strength or editor settings.")
        shots = variations(variation_mode, custom_shots) if batch_size > 1 else ()
        if reference_mode not in ("auto", "whole_image", "person_inpaint"):
            raise ValueError("ZIT Carousel: unknown reference_mode.")
        if reference_mode == "person_inpaint" and variation_mode in ("camera", "mixed"):
            raise ValueError("ZIT Carousel: person_inpaint needs a fixed camera. Use poses/custom or whole_image for camera changes.")
        backend = "pose_controlnet" if variation_backend == "auto" and variation_mode == "poses" else "reference_edit"
        needs_geometry = backend == "pose_controlnet" or reference_mode == "person_inpaint"
        if batch_size > 1 and needs_geometry and vision_model == "off":
            raise ValueError("ZIT Carousel: enable vision_model for person/head geometry, or use reference_edit with whole_image.")
        selected_model = select_vision_model(vision_model) if batch_size > 1 and vision_model != "off" and (needs_geometry or not reference_description.strip()) else None
        files = {}
        if batch_size > 1:
            selections = {"pose_controlnet": pose_controlnet} if backend == "pose_controlnet" else {"edit_model": edit_model, "edit_clip": edit_clip, "edit_vae": edit_vae}
            files = resolve_models(selections)
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
        use_inpaint = reference_mode == "person_inpaint" or (reference_mode == "auto" and backend == "pose_controlnet")
        if use_inpaint and batch_size > 1:
            if analysis is None:
                raise ValueError("ZIT Carousel: person_inpaint requires automatic QwenVL analysis. Enable vision_model.")
            mask = person_mask(reference, analysis["person_box"])

        if batch_size > 1:
            if backend == "pose_controlnet":
                pose_guide(reference, analysis, 0)  # Validate geometry before loading the patch.
                control = load_controlnet(files["pose_controlnet"])
                reference_latent = {"samples": vae.encode(reference)}
                negative = encode(clip, "")
            else:
                model, clip, vae = load_editor(files)
                negative = edit_conditioning(clip, vae, reference, "")

        output_device = comfy.model_management.intermediate_device()
        images = [reference.to(device=output_device)]
        for index in range(1, batch_size):
            comfy.model_management.throw_exception_if_processing_interrupted()
            variation = shots[(index - 1) % len(shots)]
            text = shot_prompt(variant_scene, reference_description, variation)
            shot_seed = (seed + index) % (1 << 64)
            shot_model = model
            if backend == "pose_controlnet":
                guide, target_box = pose_guide(reference, analysis, index - 1)
                mask = protect_head(person_mask(reference, target_box), analysis["head_box"]) if use_inpaint else None
                shot_model = apply_pose(model, control, vae, guide, reference, mask, control_strength)
                text += "\nKeep the original camera, head position, gaze and expression."
                positive = encode(clip, text)
                latent = {"samples": reference_latent["samples"].clone()}
                shot_steps, shot_cfg, shot_denoise = steps, cfg, denoise
            else:
                text = "Edit Picture 1. " + text
                if variation_mode == "camera":
                    text += "\nChange only the camera viewpoint/framing. Keep the person's exact pose and clothing."
                elif variation_mode == "poses":
                    text += "\nKeep the original camera viewpoint and background arrangement. Change only the pose."
                positive = edit_conditioning(clip, vae, reference, text)
                latent = EmptySD3LatentImage.execute(reference.shape[2], reference.shape[1], 1)[0]
                shot_steps, shot_cfg, shot_denoise = edit_steps, edit_cfg, 1.0
            if mask is not None:
                if backend == "reference_edit":
                    latent = {"samples": vae.encode(reference)}
                latent["noise_mask"] = mask
            result = nodes.common_ksampler(
                shot_model, shot_seed, shot_steps, shot_cfg, sampler_name, scheduler, positive, negative,
                latent, denoise=shot_denoise,
            )[0]
            decoded = vae.decode(result["samples"])
            if mask is not None:
                decoded = torch.lerp(reference.to(decoded), decoded, mask.movedim(0, -1).to(decoded))
            images.append(decoded.to(device=output_device, dtype=images[0].dtype))
            records.append({"index": index, "kind": "variation", "seed": shot_seed, "prompt": text,
                            "denoise": shot_denoise, "steps": shot_steps, "cfg": shot_cfg})
            del shot_model, result, positive, latent, decoded

        manifest = {"reference_description": reference_description, "analysis": analysis,
                    "vision_model": selected_model, "reference_mode": "person_inpaint" if mask is not None else "whole_image",
                    "backend": backend if batch_size > 1 else None, "models": files,
                    "width": reference.shape[2], "height": reference.shape[1],
                    "use_reference_resolution": use_reference_resolution, "shots": records}
        return torch.cat(images, dim=0), json.dumps(manifest, ensure_ascii=False, indent=2)
