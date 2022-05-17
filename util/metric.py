# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# https://github.com/PaddlePaddle/PaddleSeg/blob/release%2F2.5/paddleseg/utils/metrics.py


import numpy as np
import sklearn.metrics as skmetrics


def calculate_area(pred: np.ndarray, label: np.ndarray, num_classes: int, ignore_index=255):
    """
    Calculate intersect, prediction and label area
    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        Tensor: The intersection area of prediction and the ground on all class.
        Tensor: The prediction area on all class.
        Tensor: The ground truth area on all class
    """
    if len(pred.shape) == 4:
        pred = np.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = np.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape,
                                                           label.shape))
    pred_area = []
    label_area = []
    intersect_area = []
    mask = label != ignore_index

    for i in range(num_classes):
        pred_i = np.logical_and(pred == i, mask)
        label_i = label == i
        intersect_i = np.logical_and(pred_i, label_i)
        pred_area.append(np.sum(pred_i.astype(int)))
        label_area.append(np.sum(label_i.astype(int)))
        intersect_area.append(np.sum(intersect_i.astype(int)))

    pred_area = np.array(pred_area)
    label_area = np.array(label_area)
    intersect_area = np.array(intersect_area)

    return intersect_area, pred_area, label_area


def mean_iou(intersect_area, pred_area, label_area):
    """
    Calculate iou.
    Args:
        intersect_area (ndarray): The intersection area of prediction and ground truth on all classes.
        pred_area (ndarray): The prediction area on all classes.
        label_area (ndarray): The ground truth area on all classes.
    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    union = pred_area + label_area - intersect_area
    class_iou = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            iou = 0
        else:
            iou = intersect_area[i] / union[i]
        class_iou.append(iou)
    miou = np.mean(class_iou)
    return np.array(class_iou), miou

def accuracy(pred: np.ndarray, label: np.ndarray, num_classes: int, ignore_index=255):
    """
    Calculate accuracy.
    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        float: accuracy on all classes.
    """
    if len(pred.shape) == 4:
        pred = np.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = np.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape,
                                                           label.shape))
    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]
    acc = skmetrics.accuracy_score(label, pred)
    return acc


def recall(pred: np.ndarray, label: np.ndarray, num_classes: int, ignore_index=255):
    """
    Calculate recall.
    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        float: recall on all classes.
    """
    if len(pred.shape) == 4:
        pred = np.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = np.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape,
                                                           label.shape))
    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]
    rec = skmetrics.recall_score(label, pred)
    return rec


def f1_score(pred: np.ndarray, label: np.ndarray, num_classes: int, ignore_index=255):
    """
    Calculate f1 score.
    Args:
        pred (Tensor): The prediction by model.
        label (Tensor): The ground truth of image.
        num_classes (int): The unique number of target classes.
        ignore_index (int): Specifies a target value that is ignored. Default: 255.
    Returns:
        float: f1 score on all classes.
    """
    if len(pred.shape) == 4:
        pred = np.squeeze(pred, axis=1)
    if len(label.shape) == 4:
        label = np.squeeze(label, axis=1)
    if not pred.shape == label.shape:
        raise ValueError('Shape of `pred` and `label should be equal, '
                         'but there are {} and {}.'.format(pred.shape,
                                                           label.shape))
    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]
    f1 = skmetrics.f1_score(label, pred)
    return f1
