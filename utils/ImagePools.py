#### import os
import glob as gl
import os
import random

from PIL import Image
# import numpy as np
from torchvision import transforms


# import natsort


# 读取完成的图像
# training.....
class ImagePools(object):
    def __init__(self, root='', trans=None, mode=False, high=False):
        super().__init__()
        self.mode = mode
        self.high = high
        self.transform = transforms.Compose(trans)
        if not self.mode:

            self.A_path = os.path.join(root, "train_photo/*")
            self.B_path = os.path.join(root, "hayao/*")
        elif self.high:
            self.A_path = os.path.join(root, "testA/*")
            self.B_path = os.path.join(root, "testB/*")
        else:
            self.A_path = os.path.join(root, "train_photo/*")
            self.B_path = os.path.join(root, "hayao/*")
        # 读取图像文件
        #         self.list_A = natsort.natsorted(sorted(make_dataset(self.A_path,1000)))
        #         self.list_B = natsort.natsorted(sorted(make_dataset(self.B_path,1000)))
        self.list_A = gl.glob(self.A_path)
        self.list_B = gl.glob(self.B_path)

    def __getitem__(self, index):
        data = {}
        # print(self.list_A)
        # print(self.A_path, self.B_path)

        try:
            # print(self.list_A)
            A_path = self.list_A[index]
            B_path = random.choice(self.list_B)
        except IndexError:
            print(index)
            # print(A_path, B_path)
        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')
        A = self.transform(A)
        B = self.transform(B)
        data.update({"A": A, "B": B})
        return A, B

    def __len__(self):
        return max(len(self.list_A), len(self.list_B))


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]