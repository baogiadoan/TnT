import torch
import numpy as np
import cv2
from pudb import set_trace
from torch.utils.data import DataLoader, Dataset
import numpy as np


device = torch.device("cuda:0")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def unnormalize_vis(x):
    x_ = x.clone().detach()
    dtype = x_.dtype
    mean = torch.as_tensor(MEAN, dtype=dtype, device=x_.device)
    std = torch.as_tensor(STD, dtype=dtype, device=x_.device)
    x_.mul_(std[:, None, None]).add_(mean[:, None, None])
    return x_


def unnormalize(x):
    dtype = x.dtype
    mean = torch.as_tensor(MEAN, dtype=dtype, device=x.device)
    std = torch.as_tensor(STD, dtype=dtype, device=x.device)
    x.mul_(std[:, None, None]).add_(mean[:, None, None])
    return x


def normalize_vis(x):
    x_ = x.clone().detach()
    dtype = x_.dtype
    mean = torch.as_tensor(MEAN, dtype=dtype, device=x_.device)
    std = torch.as_tensor(STD, dtype=dtype, device=x_.device)
    x_.sub_(mean[:, None, None]).div_(std[:, None, None])
    return x_


def normalize(x):
    dtype = x.dtype
    mean = torch.as_tensor(MEAN, dtype=dtype, device=x.device)
    std = torch.as_tensor(STD, dtype=dtype, device=x.device)
    x.sub_(mean[:, None, None]).div_(std[:, None, None])
    # x.mul_(std[:, None, None]).add_(mean[:, None, None])
    return x



def bg_remove_threshold(img, MODEL):  # image of shape 16x16x3

    #    #==================================================
    if MODEL == 'wideresnet':
        k = 7
        T = 30
    else:
        k = 3
        T = 30

    # ==================================================

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (k, k), 0)

    # perform inverse binary thresholding
    (t, maskLayer) = cv2.threshold(blur, T, 255, cv2.THRESH_BINARY)

    maskLayer = (maskLayer / 255).astype(int)
    mask = maskLayer

    return mask


def blend_test(images, flowers, MODEL):  # input  batchx3xIMG_SIZExIMG_SIZE, flower 3xPATCH_SIZExPATCH_SIZE
    
    if MODEL == 'inception':
        CORNER = 299  # for Inception V3
        MARGIN = 0
        PATCH_SIZE = 128
    elif MODEL == 'wideresnet':
        CORNER = 224
        MARGIN = 0
        PATCH_SIZE = 128
    elif MODEL == 'vgg16':
        CORNER = 224
        MARGIN = 16
        PATCH_SIZE = 96
    # convert shape to hxwx3 for opencv to process
    flowers = torch.squeeze(flowers)
    flower_np = flowers.cpu().detach().numpy()
    flower_np = np.transpose(flower_np, (1, 2, 0))
    # convert input [-1 1] to [0 255]
    flower_np = flower_np * 0.5 + 0.5  # convert to [0 1]
    flower_np = np.uint8(flower_np * 255)  # [0 255] channel RGB
    mask = bg_remove_threshold(flower_np, MODEL)

    L0 = np.sum(mask)  # count the pixels value
    mask_size = np.round(100.0 * L0 / (CORNER * CORNER), 2)


    x0 = CORNER - MARGIN - PATCH_SIZE
    x1 = x0 + PATCH_SIZE
    y0 = CORNER - MARGIN - PATCH_SIZE
    y1 = y0 + PATCH_SIZE


    # merge to image
    mask_tensor = torch.from_numpy(mask)  # 16x16
    mask_tensor = torch.unsqueeze(mask_tensor, dim=0)  # 1x16x16
    mask_tensor = torch.cat((mask_tensor, mask_tensor, mask_tensor), dim=0)  # 3x16x16
    mask_tensor = mask_tensor.type(torch.FloatTensor).to(device)
    flowers = flowers * 0.5 + 0.5  # unnormalize flowers

    images = unnormalize(images)
    for j in range(len(images)):

        images[j][:, x0:x1, y0:y1] = (
            images[j][:, x0:x1, y0:y1]
            - mask_tensor * images[j][:, x0:x1, y0:y1]
            + mask_tensor * flowers
        )

        images = torch.clamp(images, 0, 1)

    # need to normalize the images again
    images = normalize(images)


    return images, mask_size

