import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies.droid_policy import _parse_image


@dataclasses.dataclass(frozen=True)
class B601Inputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["images"]["camera1"])
        wrist = _parse_image(data["images"]["camera2"])
        prompt = data.get("prompt", "")
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")

        result = {
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": np.zeros_like(base),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": np.asarray(data["state"], dtype=np.float32),
            "prompt": prompt,
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        return result


@dataclasses.dataclass(frozen=True)
class B601Outputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7], dtype=np.float32)}
