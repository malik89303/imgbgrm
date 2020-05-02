from PIL import Image
from pathlib import Path
import numpy as np
import cv2
from array import array
import math as math
import os
import sys


def createmask(filename):
    image = Image.open(""+filename+".png")
    pixels = image.convert('RGBA')
    width, height = image.size
    for i in range(width):
        for j in range(height):
            r, g, b, a = pixels.getpixel((i, j))
            if a == 0:
                pixels.putpixel((i, j), (0, 0, 0, 255))
            else:
                pixels.putpixel((i, j), (193, 129, 129, 255))
    pixels = pixels.convert('RGB')
    pixels.save("a"+filename+".png")
