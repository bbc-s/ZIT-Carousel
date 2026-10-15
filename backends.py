import folder_paths
import nodes
from comfy_extras.nodes_model_patch import ModelPatchLoader, ZImageFunControlnet
from comfy_extras.nodes_qwen import TextEncodeQwenImageEditPlus


MODEL_FILES = {
    "pose_controlnet": ("model_patches", "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors"),
    "edit_model": ("diffusion_models", "qwen_image_edit_2511_fp8mixed.safetensors"),
    "edit_clip": ("text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    "edit_vae": ("vae", "qwen_image_vae.safetensors"),
}


def model_choices(key):
    folder, _ = MODEL_FILES[key]
    return ["auto", *folder_paths.get_filename_list(folder)]


def resolve_models(selections):
    resolved = {}
    missing = []
    for key, selection in selections.items():
        folder, default = MODEL_FILES[key]
        name = default if selection == "auto" else selection
        if name not in folder_paths.get_filename_list(folder):
            missing.append(f"models/{folder}/{name}")
            continue
        folder_paths.get_full_path_or_raise(folder, name)
        resolved[key] = name
    if missing:
        raise ValueError("ZIT Carousel: install/select these local weights and restart ComfyUI: " + "; ".join(missing))
    return resolved


def load_editor(files):
    model = nodes.UNETLoader().load_unet(files["edit_model"], "default")[0]
    clip = nodes.CLIPLoader().load_clip(files["edit_clip"], type="qwen_image")[0]
    vae = nodes.VAELoader().load_vae(files["edit_vae"])[0]
    return model, clip, vae


def edit_conditioning(clip, vae, reference, text):
    return TextEncodeQwenImageEditPlus.execute(clip, text, vae=vae, image1=reference)[0]


def load_controlnet(name):
    return ModelPatchLoader().load_model_patch(name)[0]


def apply_pose(model, patch, vae, guide, reference, mask, strength):
    return ZImageFunControlnet().diffsynth_controlnet(
        model, patch, vae, image=guide, strength=strength,
        inpaint_image=reference if mask is not None else None, mask=mask,
    )[0]
