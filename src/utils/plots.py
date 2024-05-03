import numpy as np
import matplotlib.colors as mcolors


named_colors = list(mcolors.CSS4_COLORS.keys())
named_colors.remove('black')
named_colors.insert(0, 'black')
colors = [[int(mcolors.to_rgb(color)[0]*255), int(mcolors.to_rgb(color)[1]*255), int(mcolors.to_rgb(color)[2]*255)]
          for color in named_colors]


def norm_tensor_to_original_im(norm_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    norm_array = norm_tensor.detach().cpu().numpy().transpose((1, 2, 0))
    reverted_img = (norm_array * std + mean) * 255
    im = reverted_img.astype("uint8")
    return im


def tensors_to_images(tensor_ims):
    ims = []
    for tensor_im in tensor_ims:
        im = norm_tensor_to_original_im(tensor_im)
        ims.append(im)
    return ims
