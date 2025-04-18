# check if there is pre-trained model for img2pano.
# or just simply stretch and project.
# multi-image stiching to panorama https://github.com/davidmasek/image_stitching

import cv2
import numpy as np

def equirectangular_to_perspective(img, fov_deg, theta_deg, phi_deg, height, width):
    """
    Converts an equirectangular image to a perspective view.

    :param img: Input equirectangular image.
    :param fov_deg: Field of view in degrees.
    :param theta_deg: Yaw angle in degrees.
    :param phi_deg: Pitch angle in degrees.
    :param height: Output image height.
    :param width: Output image width.
    :return: Perspective view image.
    """
    fov = np.deg2rad(fov_deg)
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)

    # Create coordinate grid
    x = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), width)
    y = np.linspace(-np.tan(fov / 2), np.tan(fov / 2), height)
    x, y = np.meshgrid(x, -y)

    z = np.ones_like(x)

    # Normalize
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= norm
    y /= norm
    z /= norm

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi), np.cos(phi)]
    ])

    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    direction = np.stack((x, y, z), axis=-1)
    direction = direction @ Rx.T @ Ry.T

    lon = np.arctan2(direction[..., 0], direction[..., 2])
    lat = np.arcsin(direction[..., 1])

    # Normalize to image coordinates
    equi_h, equi_w = img.shape[:2]
    u = (lon / np.pi + 1) / 2 * equi_w
    v = (lat / (0.5 * np.pi) + 0.5) * equi_h

    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    # Remap the image
    perspective = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return perspective