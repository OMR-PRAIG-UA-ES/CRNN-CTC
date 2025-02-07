import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F


NUM_CHANNELS = 1
IMG_HEIGHT = 128


def apply_random_filter(x: np.ndarray) -> np.ndarray:
    # Randomly apply different transformations

    if random.random() > 0.5:
        # Rotate
        x = random_rotation(x)

    if random.random() > 0.5:
        # Contrast change
        x = random_contrast(x)

    if random.random() > 0.5:
        # Erosion
        x = random_erosion(x)

    if random.random() > 0.5:
        # Brightness adjustment
        x = random_brightness(x)

    return x


def random_rotation(image: np.ndarray) -> np.ndarray:
    angle = random.uniform(-3, 3)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR
    )
    return rotated_image


def random_contrast(image: np.ndarray) -> np.ndarray:
    alpha = random.uniform(0.5, 1.6)
    beta = random.uniform(-20, 20)
    contrast_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_adjusted


def random_erosion(image: np.ndarray) -> np.ndarray:
    kernel_size = random.randint(3, 7)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    iterations = random.randint(1, 3)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    return eroded_image


def random_brightness(image: np.ndarray) -> np.ndarray:
    brightness_factor = random.uniform(0.4, 1.5)
    brightness_adjusted = cv2.convertScaleAbs(image, alpha=brightness_factor, beta=0)
    return brightness_adjusted


def shrink_image(image):
    return image


def get_image_from_file(path: str, split: str) -> np.ndarray:
    import os

    if os.path.exists(path):
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # Resize height to IMG_HEIGHT
        # resize mantains aspect ratio
        width = int(float(IMG_HEIGHT * x.shape[1]) / x.shape[0])
        x = cv2.resize(x, (width, IMG_HEIGHT))

        # apply random filter
        if split != "test" and split != "predict":
            if random.random() > 0.4:
                x = apply_random_filter(x)
        else:
            # x = shrink_image(x)
            pass

        # Normalize
        x = x.astype(np.float32)
        x = x / 255.0
    else:
        with open("missing_files.txt", "a") as f:
            f.write(f"{path}\n")
        # print(f"File {path} not found.")
        x = np.zeros((IMG_HEIGHT, 1))

    return x


# @MEMORY.cache
def preprocess_image(path: str, split: str) -> torch.Tensor:
    x = get_image_from_file(path=path, split=split)
    x = np.expand_dims(x, axis=0)
    x = torch.from_numpy(x)
    return x


def pad_batch_images(x):
    max_width = max(x, key=lambda sample: sample.shape[2]).shape[2]
    x = torch.stack([F.pad(i, pad=(0, max_width - i.shape[2])) for i in x], dim=0)
    return x


def pad_batch_transcripts(x, dtype=torch.int32):
    max_length = max(x, key=lambda sample: sample.shape[0]).shape[0]
    x = torch.stack([F.pad(i, pad=(0, max_length - i.shape[0])) for i in x], dim=0)
    x = x.type(dtype=dtype)
    return x


def ctc_batch_preparation(batch):
    x, xl, y, yl = zip(*batch)
    # Zero-pad images to maximum batch image width
    x = pad_batch_images(x)
    xl = torch.tensor(xl, dtype=torch.int32)
    # Zero-pad transcripts to maximum batch transcript length
    y = pad_batch_transcripts(y)
    yl = torch.tensor(yl, dtype=torch.int32)
    return x, xl, y, yl
