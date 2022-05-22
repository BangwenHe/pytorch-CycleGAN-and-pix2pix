import glob
import json
import os
from collections import OrderedDict

import numpy as np
from PIL import Image

from util.metric import mean_iou, calculate_area, f1_score


if __name__ == "__main__":
    image_dir = "results/edge2mask_pix2pix/val_latest/images"

    preds_image_path = sorted(glob.glob(os.path.join(image_dir, "*_fake_B.png")))
    targets_image_path = sorted(glob.glob(os.path.join(image_dir, "*_real_B.png")))

    pred = [np.array(Image.open(i)) for i in preds_image_path]
    target = [np.array(Image.open(i)) for i in targets_image_path]

    print(len(pred), pred[0].shape)
    intersect_area, pred_area, label_area = calculate_area(pred, target, 2)
    class_iou, miou = mean_iou(intersect_area, pred_area, label_area)
    f_measure = f1_score(pred, target, 2)

    metrics = OrderedDict({'miou': miou, 'f_measure': f_measure})
    for i in range(2):
        metrics[f'class_{i}_iou'] = class_iou[i]
