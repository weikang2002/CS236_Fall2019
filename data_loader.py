import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import skimage, scipy, imageio
import os

class DataLoader():
    """Loader class for loading images"""
    def __init__(self, dataset_name, image_res=(512, 512)):
        """init data"""
        self.dataset_name = dataset_name
        self.image_res = image_res

    def image_read(self, path):
        """wrapper to load an image"""
        return imageio.imread(path, pilmode='RGB').astype(np.float)

    def load_image(self, img_file, is_testing=False):
        """load single image"""
        img_path = './datasets/{}/{}'.format(self.dataset_name, img_file)
        img = self.image_read(img_path)

        h, w = self.image_res
        low_h, low_w = h/4, w/4

        img_hr = skimage.transform.resize(img, self.image_res)
        img_lr = skimage.transform.resize(img, (low_h, low_w))

        if not is_testing and np.random.random() < 0.5:
            img_hr = np.fliplr(img_hr)
            img_lr = np.fliplr(img_lr)

        base_name = os.path.basename(img_path)
        img_name = os.path.splitext(base_name)[0]

        return img_hr, img_lr, img_name

    def load_data(self, batch_size=1, is_testing=False):
        """Load batch data"""

        path = glob('./datasets/{}/*'.format(self.dataset_name))
        batch_images = np.random.choice(path, size=batch_size)

        images_hr = []
        images_lr = []
        images_name = []
        for img_path in batch_images:
            img_file = os.path.basename(img_path)
            img_hr, img_lr, img_name = self.load_image(img_file, is_testing)
            images_hr.append(img_hr)
            images_lr.append(img_lr)
            images_name.append(img_name)

        # convert to np array, and normalize it
        images_hr = np.array(images_hr) / 127.5 - 1.
        images_lr = np.array(images_lr) / 127.5 - 1.

        return images_hr, images_lr, images_name
