import random
import torch.utils.data as data
from PIL import Image
import numpy as np
from . import preprocess

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    path_prefix = path.split('.')[0]
    path1 = path_prefix + '_exception_assign_minus_1.npy'
    path2 = path_prefix + '.npy'
    path3 = path_prefix + '.pfm'
    import os.path as ospath
    if ospath.exists(path1):
        return np.load(path1)
    else:
        if ospath.exists(path2):
            data = np.load(path2)
        else:
            from readpfm import readPFM
            data, _ = readPFM(path3)
            np.save(path2, data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if j - data[i][j] < 0:
                    data[i][j] = -1
        np.save(path1, data)
        return data


class myImageFloder(data.Dataset):
    def __init__(self,
                 left,
                 right,
                 left_disparity,
                 training,
                 loader=default_loader,
                 dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size
            left_img = left_img.crop((w - 960, h - 544, w, h))
            right_img = right_img.crop((w - 960, h - 544, w, h))
            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL, left, disp_L.split(
                '.')[0] + '_exception_assign_minus_1.npy'

    def __len__(self):
        return len(self.left)