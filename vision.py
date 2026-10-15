import json
import math
from pathlib import Path

import folder_paths
import nodes


ANALYSIS_PROMPT = '''Analyze this reference photograph for a consistent photo series.
Return ONLY a JSON object with these keys:
"person": detailed visible facial appearance, hair, skin appearance and body proportions;
"clothing": every visible garment, its color, material, cut, pattern, footwear and accessories;
"environment": location, furniture, objects, colors, materials and their arrangement;
"lighting": direction, softness, shadows, time of day and color;
"style": photographic/rendering style and palette;
"action": ONLY the coarse activity, such as sitting on the bench or standing in the studio;
"person_box": [left, top, right, bottom], the bounding box of the entire visible person,
including hair, clothing, arms, legs and shoes. Use INTEGER coordinates on a 0..1000
scale relative to the image width/height (not pixels or decimal fractions).
Each description must be a nonempty English string. Describe only visible details.
Do not identify the person or invent hidden features. Do not include exact arm positions,
head orientation, gaze, expression, camera angle or framing in the descriptions: these
will be changed later. If a detail is not visible say so. No markdown or commentary.'''
DETAIL_FIELDS = ("person", "clothing", "environment", "lighting", "style", "action")


def local_vision_models():
    node = nodes.NODE_CLASS_MAPPINGS.get("AILab_QwenVL")
    if node is None:
        return {}
    names = node.INPUT_TYPES()["required"]["model_name"][0]
    roots = folder_paths.get_folder_paths("LLM") if "LLM" in folder_paths.folder_names_and_paths else []
    root = (Path(roots[0]) if roots else Path(folder_paths.models_dir) / "LLM") / "Qwen-VL"
    models = {}
    for name in names:
        if Path(name).name != name or name in (".", ".."):
            continue
        target = (root / name).resolve()
        if not target.is_relative_to(root.resolve()):
            continue
        if (target / "config.json").is_file() and any(target.glob("*.safetensors")):
            models[name] = target
    return models


def select_vision_model(selection):
    if "AILab_QwenVL" not in nodes.NODE_CLASS_MAPPINGS:
        raise ValueError("ZIT Carousel: automatic analysis requires the ComfyUI-QwenVL extension to load successfully.")
    models = local_vision_models()
    if selection == "auto" and models:
        return min(models, key=lambda name: sum(p.stat().st_size for p in models[name].glob("*.safetensors")))
    if selection in models:
        return selection
    raise ValueError("ZIT Carousel: no selected local QwenVL model. Install a complete supported model in models/LLM/Qwen-VL/<model-name>, restart ComfyUI, or supply reference_description. Carousel does not download models.")


def parse_analysis(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.partition("\n")[2].removesuffix("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("ZIT Carousel: QwenVL did not return valid reference JSON. Inspect the QwenVL model output.") from error
    if not isinstance(data, dict):
        raise ValueError("ZIT Carousel: QwenVL reference analysis must be a JSON object.")
    details = {}
    for key in DETAIL_FIELDS:
        value = data.get(key)
        if isinstance(value, (dict, list)) and value:
            value = json.dumps(value, ensure_ascii=False)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"ZIT Carousel: QwenVL reference JSON is missing a {key} description.")
        details[key] = value.strip()
    box = data.get("person_box")
    if not isinstance(box, list) or len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
        raise ValueError("ZIT Carousel: QwenVL must return person_box as four numbers on the 0..1000 scale.")
    left, top, right, bottom = box
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000) or max(box) <= 1:
        raise ValueError("ZIT Carousel: QwenVL person_box must be ordered left, top, right, bottom on the 0..1000 scale.")
    return details | {"person_box": [value / 1000 for value in box]}


def analyze_reference(image, model_name, seed):
    node = nodes.NODE_CLASS_MAPPINGS["AILab_QwenVL"]()
    result = node.process(
        model_name=model_name, quantization="None (FP16)", attention_mode="auto",
        preset_prompt="", custom_prompt=ANALYSIS_PROMPT, max_tokens=1024,
        keep_model_loaded=False, seed=seed % ((1 << 32) - 1) + 1, image=image,
    )
    # AILab_QwenVL documents a single STRING output in a tuple.
    return parse_analysis(result[0])
