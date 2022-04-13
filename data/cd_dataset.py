"""
加载压缩域信息数据集
"""
import os
import os.path as osp

import numpy as np

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


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


class CDDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        voc2012_dir = osp.join(opt.dataroot, 'VOCdevkit', 'VOC2012')
        voc2021_train_file = osp.join(voc2012_dir, 'ImageSets', 'Segmentation', 'train.txt')
        with open(voc2021_train_file, 'r') as f:
            filenames = f.readlines()
            image_paths = [osp.join(voc2012_dir, 'JPEGImages', line.strip() + '.jpg') for line in filenames]
            cd_paths = [osp.join(voc2012_dir, 'CompressedDomainData', line.strip() + '.txt') for line in filenames]

        assert len(image_paths) == len(cd_paths), f'len(image_paths)={len(image_paths)}, len(cd_paths)={len(cd_paths)}'

        files_idxes = np.random.choice(len(image_paths), self.opt.max_dataset_size, replace=False)
        self.image_paths = [image_paths[i] for i in files_idxes]
        self.cd_paths = [cd_paths[i] for i in files_idxes]

        self.transform = get_transform(opt)

    def __getitem__(self, index):
        path = self.image_paths[index]
        cd_path = self.cd_paths[index]
        image = Image.open(path).convert('RGB')
        cd = Image.fromarray(load_cdd(cd_path, image.width, image.height))

        if image.width > image.height:
            image = image.rotate(90, expand=True)
            cd = cd.rotate(90, expand=True)
        
        data_A = self.transform(image)
        data_B = self.transform(cd)

        # return {'A': data_A, 'B': data_B, 'A_image': image, 'B_image': cd}
        return {'A': data_A, 'B': data_B}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)
