from collections import deque

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _normalize_min_max(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


class Resize:
    def __init__(self, config):
        if config.get("grey_scale", True):
            self.transform = transforms.Compose(
                [
                    transforms.Resize((config["size_x"], config["size_y"])),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor(),
                    transforms.Lambda(_normalize_min_max),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((config["size_x"], config["size_y"])),
                    transforms.ToTensor(),
                    transforms.Lambda(_normalize_min_max),
                ]
            )

    def reset(self):
        pass

    def __call__(self, sample):
        img, eye_gazes, action = sample

        # ✅ Handle different input types safely
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)  # numpy → PIL

        if (
            isinstance(img, torch.Tensor)
            and img.ndim == 3
            and img.shape[0] not in [1, 3]
        ):
            # likely (H, W, C) → convert
            img = img.permute(2, 0, 1)

        # PIL images pass through untouched

        img = self.transform(img)

        return img, eye_gazes, action


class Stack:
    def __init__(self, config):
        self.stack_len = config["stack_length"]
        self.stack = deque(maxlen=self.stack_len)
        self.config = config

    def reset(self):
        self.stack.clear()

    def __call__(self, sample):
        img, action = sample
        if len(self.stack) < self.stack_len:
            while len(self.stack) < self.stack_len:
                self.stack.append(img)
        else:
            self.stack.append(img)
        stacked = torch.cat(list(self.stack), dim=0)

        return stacked


class StackWithLabels:
    """
    Stacks images, gaze, and actions in temporal order.

    Output ordering:
        [t-3, t-2, t-1, t]

    Therefore:
        stacked_action[-1] == action for newest frame t
    """

    def __init__(self, config):
        self.stack_len = config["stack_length"]
        self.config = config

        self.img_stack = deque(maxlen=self.stack_len)
        self.gaze_stack = deque(maxlen=self.stack_len)
        self.action_stack = deque(maxlen=self.stack_len)

    def reset(self):
        self.img_stack.clear()
        self.gaze_stack.clear()
        self.action_stack.clear()

    def __call__(self, sample):
        img, eye_gaze, action = sample
        if len(self.img_stack) < self.stack_len:
            while len(self.img_stack) < self.stack_len:
                self.img_stack.append(img)
                self.gaze_stack.append(eye_gaze)
                self.action_stack.append(action)

        else:
            # IMPORTANT:
            # Everything gets appended in the SAME direction.
            #
            # deque(maxlen=N) automatically removes the
            # oldest item from the left.
            self.img_stack.append(img)
            self.gaze_stack.append(eye_gaze)
            self.action_stack.append(action)
        stacked_img = torch.cat(
            list(self.img_stack),
            dim=0,
        )
        stacked_action = torch.tensor(
            list(self.action_stack),
            dtype=torch.long,
        )
        return (
            stacked_img,
            stacked_action,
        )


class ComposePreprocessor:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors

    def reset(self):
        for p in self.preprocessors:
            p.reset()

    def __call__(self, sample):
        for p in self.preprocessors:
            sample = p(sample)

        return sample
