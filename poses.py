import numpy as np
import torch
from PIL import Image, ImageDraw


# OpenPose body ordering; target templates, not detected source keypoints.
LIMBS = ((1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
         (0, 14), (14, 16), (0, 15), (15, 17))
COLORS = ((255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
          (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
          (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
          (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
          (255, 0, 170), (255, 0, 85))
ARMS = (
    ((-.35, .50), (-.27, .68), (.35, .50), (.27, .68)),
    ((-.45, .40), (-.28, .48), (.45, .40), (.28, .48)),
    ((-.46, .12), (-.28, -.10), (.46, .12), (.28, -.10)),
    ((-.53, .29), (-.77, .30), (.53, .29), (.77, .30)),
    ((-.46, .40), (-.28, .48), (.46, .20), (.35, .03)),
    ((-.35, .45), (-.38, .68), (.46, .12), (.28, -.10)),
)
POSES = (
    "both hands resting on the thighs, elbows slightly bent",
    "both hands on the hips, elbows pointing outward",
    "both arms raised overhead with bent elbows",
    "both arms stretched out to the sides",
    "left hand raised beside the head, right hand on the hip",
    "left arm raised overhead, right arm hanging down",
)


def pose_guide(reference, analysis, index):
    if analysis.get("posture") not in ("standing", "seated") or "head_box" not in analysis:
        raise ValueError("ZIT Carousel: pose control needs head_box and standing/seated posture from QwenVL. Use reference_edit for other poses.")
    h, w = reference.shape[1:3]
    left, top, right, bottom = analysis["person_box"]
    hl, ht, hr, hb = analysis["head_box"]
    cx = (hl + hr) / 2
    neck = hb
    body_height = bottom - neck
    if body_height < .15:
        raise ValueError("ZIT Carousel: too little visible body for pose templates. Use reference_edit.")
    span = max((hr - hl) * 2.5, (right - left) * .7)
    arms = ARMS[index % len(ARMS)]
    points = [(cx, (ht + hb) / 2), (cx, neck)]
    points += [(cx - .28 * span, neck + .08 * body_height)]
    points += [(cx + x * span, neck + y * body_height) for x, y in arms[:2]]
    points += [(cx + .28 * span, neck + .08 * body_height)]
    points += [(cx + x * span, neck + y * body_height) for x, y in arms[2:]]
    knee = .66 if analysis["posture"] == "standing" else .60
    knee_spread = .18 if analysis["posture"] == "standing" else .36
    for side in (-1, 1):
        points += [(cx + side * .18 * span, neck + .48 * body_height),
                   (cx + side * knee_spread * span, neck + knee * body_height),
                   (cx + side * knee_spread * span, neck + .96 * body_height)]
    points += [(cx - (hr - hl) * .16, (ht + hb) / 2 - (hb - ht) * .1),
               (cx + (hr - hl) * .16, (ht + hb) / 2 - (hb - ht) * .1),
               (hl, (ht + hb) / 2), (hr, (ht + hb) / 2)]
    points = [(min(.99, max(.01, x)), min(.99, max(.01, y))) for x, y in points]
    canvas = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(canvas)
    radius = max(2, round(min(w, h) / 256))
    pixels = [(round(x * (w - 1)), round(y * (h - 1))) for x, y in points]
    for i, (a, b) in enumerate(LIMBS):
        draw.line((pixels[a], pixels[b]), fill=tuple(round(c * .6) for c in COLORS[i]), width=radius * 2)
    for point, color in zip(pixels, COLORS):
        x, y = point
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    box = [min(left, min(x for x, _ in points)), min(top, min(y for _, y in points)),
           max(right, max(x for x, _ in points)), max(bottom, max(y for _, y in points))]
    return torch.from_numpy(np.array(canvas).astype(np.float32) / 255).unsqueeze(0), box


def protect_head(mask, box):
    h, w = mask.shape[-2:]
    left, top, right, bottom = box
    padding = .02
    mask[:, max(0, int((top - padding) * h)):min(h, int((bottom + padding) * h) + 1),
         max(0, int((left - padding) * w)):min(w, int((right + padding) * w) + 1)] = 0
    return mask
