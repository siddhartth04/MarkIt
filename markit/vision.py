"""Florence-2 loading and inference, plus the MediaPipe pose model.

Importing this module loads the model — that is deliberate: everything else
depends on it being ready, and loading twice would double the memory use.
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

from .config import FLORENCE_MODEL

def _fixed_get_imports(filename):
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

print("Loading Florence-2 (first run downloads weights)...")
with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        FLORENCE_MODEL, trust_remote_code=True, torch_dtype=torch.float32
    ).to("cpu").eval()
print("Florence-2 ready.")

def florence(image_pil, task_prompt, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + text_input
    inputs = processor(text=prompt, images=image_pil, return_tensors="pt")
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
            max_new_tokens=256, num_beams=1, do_sample=False,
        )
    text = processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        text, task=task_prompt, image_size=(image_pil.width, image_pil.height))


def normalize_prompt(text):
    """Turn any separator style into Florence-2's expected 'a. b. c.' form.

    Florence-2 splits the watch list on periods, and stray spacing corrupts the
    phrases: 'person . phone' becomes 'person ' and ' phone', which measurably
    degrades grounding (it returned a spurious third box in testing). Users
    reasonably type commas, 'and', or spaced periods, so accept all of them.
    """
    import re
    parts = [p.strip() for p in re.split(r"[.,;]|\band\b", text)]
    parts = [p for p in parts if p]
    return ". ".join(parts) + "." if parts else ""

# ----------------------------------------------------------------------------
# MediaPipe Pose
# ----------------------------------------------------------------------------

# --------------------------------------------------------------------------
# MediaPipe Pose
# --------------------------------------------------------------------------
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
