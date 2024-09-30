import torch
import cv2
import random
import numpy as np
from PIL import Image
from torchvision import transforms as T


v_transforms = T.Compose([
    T.Resize(432),
    T.CenterCrop((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])])

def find_point(id_x, id_y, ids):
    sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
    return [id_x[sel_id], id_y[sel_id]]

def crop_funduimg(imgpath, thres=10):
    img = cv2.imread(imgpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, thres, 255, cv2.THRESH_BINARY)[1]

    pixel = 255
    pert = 0
    inds_y, inds_x = np.where(mask == pixel)  # 搜寻特定像素的点

    # Find extreme points
    result = np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x) + pert)),  # left
                       find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x) - pert)),  # right
                       find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y) + pert)),  # top
                       find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y) - pert))  # bottom
                       ])

    x0, y0, x1, y1 = min(result[:, 0]), min(result[:, 1]), max(result[:, 0]), max(result[:, 1])  # 极值点xy值
    bbox = (x0, y0, x1, y1)

    img = Image.open(imgpath)

    img_crop = img.crop(bbox)

    return img_crop

def val_transforms(imgpath):
    # img = crop_funduimg(imgpath, thres=0)
    img = Image.open(imgpath).convert('RGB')
    output = v_transforms(img)

    return output
