"""Image handling for the detection stage.

ROS image decoding lives here rather than in geometry/ because detection is what
consumes CCTV frames. tools/dump_cctv.py imports it too, so the recorder and the
agent decode frames exactly the same way.
"""

import numpy as np

# Canonical catalogue spellings used by both the NLU/grounding seam and detector
# routing.  The NLU already fixes these common variants; keeping the same small,
# context-independent map here is a final defence for callers that bypass it.  Do not
# add ambiguous semantic guesses (for example ``table`` -> ``picnic_table``).
DETECTION_LABEL_ALIASES = {
    "fire_hydrant": "hydrant",
    "fire hydrant": "hydrant",
    "drink_machine": "vending_machine",
    "drink machine": "vending_machine",
    "park_bench": "bench",
    "park bench": "bench",
    "bike": "bicycle",
    "pencil_case": "pencilcase",
    "pencil case": "pencilcase",
    "juice_box": "juice",
    "juice box": "juice",
    "tissue_pack": "tissue",
    "tissue pack": "tissue",
}


def normalize_detection_label(label):
    """Return a stable detector catalogue label, preserving ``None``/empty values."""
    if not label:
        return label
    value = str(label).strip().lower()
    return DETECTION_LABEL_ALIASES.get(value, value.replace(" ", "_"))

# OWLv2 uses natural-language prompts rather than canonical catalogue IDs.
# This table covers known project labels; unknown labels receive a generated prompt.
# YOLO handles its supported labels without using this prompt table.
DETECT_PROMPT = {
    # Shape/material descriptions improve matching for some small objects but can
    # also match poles or table legs. Spatial grounding and confidence floors help
    # reject those distractors; the prompts alone do not uniquely identify a class.
    "mug": "a mug",
    "tumbler": "a water bottle",
    "umbrella": "a long thin object",
    "sunblock": "a plastic bottle",
    "pencilcase": "a pencil case",
    "cola_can": "a soda can",
    # OWLv2 receives the same generic person prompt for all pose labels. Its boxes
    # therefore do not independently establish pose; requested label keys are retained
    # in the output. The YOLO path uses the trained pose-specific class heads.
    "person": "a person",                    # NLU may still emit the plain label
    "person_sitting": "a person",
    "person_lying_down": "a person",
    "person_prone": "a person",
    "person_bending": "a person",
    "person_crouching": "a person",
    "person_reaching": "a person",
    "person_reclining": "a person",
    "person_walking": "a person",
    "person_walking_away": "a person",
    # landmarks
    "picnic_table": "a picnic table",
    "bench": "a bench",
    "hydrant": "a fire hydrant",
    "vending_machine": "a vending machine",
    "trash_can": "a trash can",
    "bicycle": "a bicycle",
    "taxi": "a taxi",
    "police_car": "a police car",
    # Vehicle variant numbers are not useful natural-language descriptions.
    # Variants sharing a prompt can produce duplicate OWLv2 boxes under different
    # labels; this fallback cannot reliably distinguish those variants.
    "postbox": "a postbox",
    "kick_scooter": "a kick scooter",
    "normal_car_1": "a car",
    "normal_car_2": "a car",
    "sports_car": "a sports car",
    "sports_car_2": "a sports car",
    "suv": "an SUV",
}


# Extra OWLv2 confidence floors for ambiguous shape-description prompts.
# The umbrella prompt also matches poles and table legs; its higher floor limits
# weak distractors without raising the global threshold for other classes.
# These floors do not affect the YOLO path.
DETECT_FLOOR = {
    "umbrella": 0.12,   # "a long thin object" -- also fires on poles and table legs
}


def image_bytes_to_numpy(data, height, width, encoding):
    """Raw ROS image buffer -> HxWx3 uint8 RGB.

    Takes the raw fields rather than a sensor_msgs/Image so this module stays
    importable without ROS (the recorder runs on the platform machine, but the
    same decoding is useful when replaying a dump on a laptop).
    """
    buf = np.frombuffer(data, dtype=np.uint8)
    enc = encoding.lower()
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(height, width, 3)
        return img[:, :, ::-1] if enc == "bgr8" else img
    if enc in ("rgba8", "bgra8"):
        img = buf.reshape(height, width, 4)[:, :, :3]
        return img[:, :, ::-1] if enc == "bgra8" else img
    if enc == "mono8":
        return np.repeat(buf.reshape(height, width, 1), 3, axis=2)
    raise ValueError(f"unsupported image encoding: {encoding}")
