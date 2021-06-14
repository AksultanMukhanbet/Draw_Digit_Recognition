from flask import Flask
from flask import request
import numpy as np
from PIL import Image
from PIL import ImageOps
import base64


def crop_to_digit(img):
    """
    Crop image to the exact size of the drawn digit
    """

    img_size = img.size[0]

    crop_left = img_size
    crop_right = 0
    crop_up = img_size
    crop_down = 0

    img_data = img.load()
    for x in range(img_size):
        for y in range(img_size):
            if img_data[(x, y)] != 0:
                crop_left = min(x, crop_left)
                crop_right = max(x, crop_right)
                crop_up = min(y, crop_up)
                crop_down = max(y, crop_down)

    return img.crop((crop_left, crop_up, crop_right + 1, crop_down + 1))


def expand_to_square(img):

    if img.size[0] > img.size[1]:
        expand_amount = (img.size[0] - img.size[1]) // 2
        cropped = ImageOps.expand(img, border=(0, expand_amount, 0, expand_amount))
    else:
        expand_amount = (img.size[1] - img.size[0]) // 2
        cropped = ImageOps.expand(img, border=(expand_amount, 0, expand_amount, 0))

    return cropped


def scale_to_mnist(img):
    scaled = img.resize([20, 20], Image.BICUBIC)
    return ImageOps.expand(scaled, border=4)


def center_digit(img):

    img_size = img.size[0]
    img_data = img.load()
    m = np.zeros((img_size, img_size))

    for x in range(img_size):
        for y in range(img_size):
            m[x, y] = img_data[(x, y)] != 0

    s = np.sum(np.sum(m))

    if s == 0:
        return None

    m = m / s

    dx = np.sum(m, 1)
    dy = np.sum(m, 0)

    cx = np.sum(dx * np.arange(img_size))
    cy = np.sum(dy * np.arange(img_size))

    middle = img_size / 2
    offset_x = cx - middle
    offset_y = cy - middle

    a = 1
    b = 0
    c = round(offset_x)  # left/right (i.e. 5/-5)
    d = 0
    e = 1
    f = round(offset_y)  # up/down (i.e. 5/-5)

    return img.transform(img.size, Image.AFFINE, (a, b, c, d, e, f))


def post_data_to_image(image_data, img_size):
    """
    Converts data from the post form into a PIL Image
    """

    rle_bytes = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
    right_nibbles = rle_bytes & 0xf
    left_nibbles = (rle_bytes >> 4) & 0xf
    nibbles = np.zeros(rle_bytes.size * 2, dtype=np.uint8)
    nibbles[0::2] = left_nibbles
    nibbles[1::2] = right_nibbles

    comma_indexes = [-1] + [i for i, v in enumerate(nibbles) if v == 0] + [len(nibbles)]
    rle_nums = ["".join(chr(44 if b == 0 else b + 47) for b in nibbles[comma_indexes[i] + 1:comma_indexes[i + 1]]) for i in range(len(comma_indexes) - 1)]
    pixel_nums = np.zeros(img_size * img_size, dtype=np.uint8)
    pixel_value = 0
    index = 0

    for i, length in enumerate(rle_nums):
        for j in range(int(length)):
            pixel_nums[index] = pixel_value
            index += 1
        pixel_value = 255 if pixel_value == 0 else 0

    reshaped = pixel_nums.reshape((img_size, img_size))
    return Image.fromarray(reshaped)


def mnistify_image(img, save_stages=False):
    """
    Converts the canvas image into an image that resembles a digit from the MNIST
    data set as closely as possible. This is broken down into 4 steps:

    1. Crop image to digit
    2. Extend image out into a square
    3. Scale the image down to 20x20, and expand edges out to 28x28
    4. Center digit within image based on Center of Mass

    :param save_stages: Saves each stage of image processing as a file if True
    """

    cropped = crop_to_digit(img)
    squared = expand_to_square(cropped)
    scaled = scale_to_mnist(squared)
    centered = center_digit(scaled)

    if save_stages:
        img.save("debug/0-canvas-img.png")
        cropped.save("debug/1-cropped.png")
        squared.save("debug/2-squared.png")
        scaled.save("debug/3-scaled.png")
        centered.save("debug/4-centered-final.png")

    return centered


def image_to_model_input(img):
    """
    Converts a 28x28 PIL image to data that can be used as input for the Keras model
    """

    model_input = np.array(img).reshape((1, 28, 28, 1))
    return model_input
