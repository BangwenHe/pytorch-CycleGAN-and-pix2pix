"""
加载压缩域信息数据集
"""
import os
import os.path as osp
from secrets import choice

import numpy as np

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageOps


def load_cdd(cdd_filepath, image_width, image_height):
    frame = np.loadtxt(cdd_filepath, delimiter=' ', dtype=int)
    width_max = np.max(frame[:, 2] + frame[:, 4])
    height_max = np.max(frame[:, 3] + frame[:, 5])
    image = np.zeros((height_max, width_max, 3), np.uint8)
    for line in frame:
        mv_y = line[0]
        mv_x = line[1]
        local_y = line[2]
        local_x = line[3]
        mb_height = line[4]
        mb_width = line[5]

        mv_len = np.sqrt(mv_y ** 2 + mv_x ** 2)
        mv_angle = np.arctan2(mv_x, mv_y)
        mv_angle = mv_angle * 180 / np.pi
        mb = int(np.log2(mb_width * mb_height))

        vec = np.array([mv_len, mv_angle, mb])
        mat = np.expand_dims(vec, axis=0)
        mat = np.repeat(mat, mb_height * mb_width, axis=0)
        mat = mat.reshape(mb_width, mb_height, 3)

        image[local_x: local_x+mb_width, local_y: local_y+mb_height] = mat
    
    # 区分横竖
    if image_height > image_width:
        image = image.transpose(1, 0, 2)
    image = image[0:image_height, 0:image_width]

    mv_len = image[..., 0]
    mv_angle = image[..., 1]
    mb = image[..., 2]

    # 着色
    if np.max(mv_len) - np.min(mv_len) > 0:
        image[..., 0] = (mv_len - np.min(mv_len)) / (np.max(mv_len) - np.min(mv_len)) * 255
    image[..., 1] = (mv_angle - -180) / (180 - np.min(mv_angle)) * 255
    image[..., 2] = (mb - 1) / (16 - np.min(mb)) * 255

    # 翻转
    image = np.flip(image, axis=1)
    return image


# Copyright @PaddleSeg
def get_pseudo_color_map(pred, color_map=None):
    """
    Get the pseudo color image.
    Args:
        pred (numpy.ndarray): the origin predicted image.
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.
    
    Returns:
        (numpy.ndarray): the pseduo image.
    """
    pred_mask = Image.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.
    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.
    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


class CDDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        voc2012_dir = osp.join(opt.dataroot, 'VOCdevkit', 'VOC2012')
        voc2021_train_file = osp.join(voc2012_dir, 'ImageSets', 'Segmentation', 'train.txt')
        with open(voc2021_train_file, 'r') as f:
            filenames = f.readlines()
            if opt.target == 'rgb':
                image_paths = [osp.join(voc2012_dir, 'JPEGImages', line.strip() + '.jpg') for line in filenames]
            elif opt.target == 'mask':
                image_paths = [osp.join(voc2012_dir, 'SegmentationClass', line.strip() + '.png') for line in filenames]
            cd_paths = [osp.join(voc2012_dir, 'CompressedDomainData', line.strip() + '.txt') for line in filenames]

        assert len(image_paths) == len(cd_paths), f'len(image_paths)={len(image_paths)}, len(cd_paths)={len(cd_paths)}'

        if len(image_paths) > self.opt.max_dataset_size:
            files_idxes = np.random.choice(len(image_paths), self.opt.max_dataset_size, replace=False)
        else:
            files_idxes = np.arange(len(image_paths))
            np.random.shuffle(files_idxes)
        self.image_paths = [image_paths[i] for i in files_idxes]
        self.cd_paths = [cd_paths[i] for i in files_idxes]

        self.transform = get_transform(opt)
        self.color_map = get_color_map_list(2)
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        print(parser)
        parser.add_argument('--crop_person', action='store_true', help='crop person')
        parser.add_argument('--target', choices=['rgb', 'mask'], default='rgb', help='train target, rgb or mask')
        return parser

    def encode_target(self, mask):
        """Encode mask to target"""
        mask = np.array(mask)
        mask[mask != 11] = 0
        mask[mask == 11] = 1
        return get_pseudo_color_map(mask, color_map=self.color_map)
    
    def crop_person(self, image, mask):
        """Crop person from image and mask"""
        mask_np = np.array(mask)
        idx = np.where(mask_np == 1)
        if len(idx[0]) == 0:
            return image, mask
        
        xmin = np.min(idx[1])
        xmax = np.max(idx[1])
        ymin = np.min(idx[0])
        ymax = np.max(idx[0])
        image_crop = image.crop((xmin, ymin, xmax, ymax))
        mask_crop = mask.crop((xmin, ymin, xmax, ymax))
        return image_crop, mask_crop

    def __getitem__(self, index):
        path = self.image_paths[index]
        cd_path = self.cd_paths[index]
        image = Image.open(path)
        cd = Image.fromarray(load_cdd(cd_path, image.width, image.height))

        if image.width > image.height:
            image = image.rotate(90, expand=True)
            cd = cd.rotate(90, expand=True)
        
        if self.opt.target == 'mask':
            image = self.encode_target(image)
        
        if self.opt.crop_person:
            assert self.opt.target == 'mask', 'crop_person only support mask target'
            cd, image = self.crop_person(cd, image)
            image = image.convert('RGB')
        
        data_A = self.transform(image)
        data_B = self.transform(cd)

        # return {'A': data_A, 'B': data_B, 'A_image': image, 'B_image': cd}
        return {'A': data_A, 'B': data_B}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
