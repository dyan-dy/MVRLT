# here we patchify envmap to patches.
# use blender for multiview snapshot with camera params. (take care of view density while sampling) (then goes to blender_render.py)
# or `pip install patchify` (only for image, then adding projection matrix info)

from patchify import patchify, unpatchify
import cv2
import numpy as np
from matplotlib import pyplot as plt

