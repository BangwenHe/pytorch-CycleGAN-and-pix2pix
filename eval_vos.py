import glob
import json
import os
import datetime

import cv2
import numpy as np
from PIL import Image

from util.metric import mean_iou, calculate_area


if __name__ == "__main__":
    result_dir = "results/eval_vos"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    pred_dir = "results/cd_vos7_cd2mask_0to255_pix2pix/test_200/images"
    pred_files = glob.glob(pred_dir + "/*fake_B*.png")
    print(f"num of pred files: {len(pred_files)}")

    label_dir = "datasets/VOS7/VOCdevkit/VOC2012/SegmentationClass"

    iou_per_image = {}

    for pred_file in pred_files:
        filename = os.path.basename(pred_file)
        label_file = os.path.join(label_dir, filename.replace("_fake_B", ""))

        pred = np.array(Image.open(pred_file))[..., 0]
        pred = cv2.threshold(pred, 245, 1, cv2.THRESH_BINARY)[1]
        label = np.array(Image.open(label_file))
        label = cv2.resize(label, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        label[label > 1] = 1

        intersect_area, pred_area, label_area = calculate_area(pred, label, num_classes=2)
        class_iou, miou = mean_iou(intersect_area, pred_area, label_area)
        iou_per_image[filename] = class_iou.tolist()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(result_dir, f"iou_per_image_{timestamp}.json"), "w") as f:
        json.dump(iou_per_image, f, indent=4)

    # calculate mean iou
    iou_per_class = {}
    for filename, iou in iou_per_image.items():
        for i, iou_per_class_i in enumerate(iou):
            if i not in iou_per_class:
                iou_per_class[i] = []
            iou_per_class[i].append(iou_per_class_i)

    iou_thresholds = [0.1, 0.25, 0.5]
    for iou_threshold in iou_thresholds:
        print(f"iou_threshold: {iou_threshold}")
        for i, iou_per_class_i in iou_per_class.items():
            iou_per_class_i = np.array(iou_per_class_i)
            iou_per_class_i = iou_per_class_i[iou_per_class_i > iou_threshold]
            print(f"class-{i} mIoU: {np.mean(iou_per_class_i)} len: {len(iou_per_class_i)}")
