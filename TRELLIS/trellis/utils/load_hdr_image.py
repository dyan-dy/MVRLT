import torch
import numpy as np
from pathlib import Path
from typing import Union

def load_hdr_image(path: Union[str, Path]) -> torch.Tensor:
    """
    Load an HDR (.hdr) or EXR (.exr) image and convert it to a float32 PyTorch tensor (1, 3, H, W).
    Supports RGB images only.

    Returns:
        torch.Tensor: Tensor of shape (1, 3, H, W), dtype float32
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".hdr":
        import imageio.v3 as iio
        img = iio.imread(str(path))  # (H, W, 3), float32
        if img.dtype != np.float32:
            img = img.astype(np.float32)

    elif suffix == ".exr":
        import OpenEXR, Imath
        if not path.exists():
            raise FileNotFoundError(f"EXR file not found: {path}")
        
        exr_file = OpenEXR.InputFile(str(path))
        header = exr_file.header()
        dw = header['dataWindow']
        H = dw.max.y - dw.min.y + 1
        W = dw.max.x - dw.min.x + 1

        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = ['R', 'G', 'B']
        img = np.stack([
            np.frombuffer(exr_file.channel(c, pt), dtype=np.float32).reshape(H, W)
            for c in channels
        ], axis=-1)  # (H, W, 3)

    else:
        raise ValueError(f"Unsupported HDR file type: {suffix}")

    # Convert to (1, 3, H, W)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    return tensor
