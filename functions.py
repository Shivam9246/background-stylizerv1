import numpy as np
import tensorflow as tf
from PIL import Image


def crop_center(image):
    """
    Crops the largest central square from an image tensor.
    Useful for making non-square images square before style transfer.
    """
    shape = image.shape
    new_shape = min(shape[1], shape[2])
    offset_y = max(shape[1] - shape[2], 0) // 2
    offset_x = max(shape[2] - shape[1], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape
    )
    return image


def load_and_preprocess_image(image: Image.Image, image_size: tuple = (256, 256)) -> tf.Tensor:
    """
    Takes a PIL Image, resizes it, normalizes pixel values to [0, 1],
    and returns a float32 tensor of shape [1, H, W, 3].

    Args:
        image: PIL.Image object (RGB)
        image_size: target (width, height) tuple

    Returns:
        tf.Tensor of shape [1, H, W, 3] with values in [0.0, 1.0]
    """
    image = image.convert("RGB")
    image = image.resize(image_size)
    image_array = np.array(image)
    image_array = image_array.astype(np.float32) / 255.0
    image_tensor = tf.constant(image_array)
    image_tensor = image_tensor[tf.newaxis, ...]  # Add batch dimension
    return image_tensor


def save_image(image_tensor: tf.Tensor, filename: str):
    """
    Saves a stylized image tensor to disk as a JPEG.

    Args:
        image_tensor: tensor of shape [1, H, W, 3] with values in [0, 1]
        filename: output file path (e.g. 'output.jpg')
    """
    image_np = image_tensor.numpy()
    if image_np.ndim == 4:
        image_np = image_np[0]  # Remove batch dimension
    image_np = (image_np * 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np)
    pil_image.save(filename)
