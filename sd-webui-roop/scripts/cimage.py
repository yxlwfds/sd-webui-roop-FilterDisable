from typing import List, Union, Dict, Set, Tuple

# from diffusers.pipelines.stable_diffusion.safety_checker import (
#    StableDiffusionSafetyChecker,
# )
# from transformers import AutoFeatureExtractor
import torch
from PIL import Image, ImageFilter
import numpy as np

# safety_model_id: str = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor: AutoFeatureExtractor = None
# safety_checker: StableDiffusionSafetyChecker = None


def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def check_image(x_image: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
    return x_image, []


def check_batch(x: torch.Tensor) -> torch.Tensor:
    return x


def convert_to_sd(img: Image) -> Image:
    return img
