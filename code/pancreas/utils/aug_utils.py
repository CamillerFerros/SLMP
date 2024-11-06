import math
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from .transform_argument import random_rot_flip, random_rotate, blur, obtain_cutmix_box

from copy import deepcopy
import h5py
import math
import numpy as np
import os
from PIL import Image
import random
from scipy.ndimage.interpolation import zoom
from scipy import ndimage
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from volumentations import *

from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.util import random_noise
from PIL import ImageFilter


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def random_blur(image, max_sigma=2.0):
    if random.random() < 0.5:
        sigma = np.random.uniform(0, max_sigma)
        return gaussian_filter(image, sigma)
    else:
        return image


def random_gamma(image, gamma_range=(0.5, 2.0)):
    gamma = np.random.uniform(*gamma_range)
    return exposure.adjust_gamma(image, gamma)


def add_gaussian_noise(image, var=0.01):
    noise = np.random.normal(0, var ** 0.5, image.shape)
    return np.clip(image + noise, 0, 1)


def augment(imgs):
    """
    INPUT: numpy array of shape num_samples*3*img_size*img_size [dtype = float32]
    OUTPUT = numpy array of same shape as input with augmented images
    """

    imgs = (imgs * 255).astype(np.uint8)
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
    # image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    aug = iaa.Sequential(
        # Define our sequence of augmentation steps that will be applied to every image.
        [
            iaa.Fliplr(0.7),  # horizontally flip 70% of all images
            # iaa.Flipud(0.2), # vertically flip 20% of all images

            # crop some of the images by 0-20% of their height/width
            sometimes(iaa.Crop(percent=(0, 0.2))),

            iaa.SomeOf((1, 5), [
                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                ),

                # color jitter the image == change brightness, contrast, hue and saturation
                iaa.AddToHueAndSaturation((-50, 50), per_channel=0.5),
                iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
            ]),
            sometimes(iaa.Grayscale(alpha=(0.0, 1.0)))
        ]
    )

    imgs = aug(images=imgs)
    imgs = (imgs / 255.).astype(np.float32)

    return imgs


def batch_augment(list_imgs, N):
    """
    INPUT:
    list_imgs =list of (numpy array of shape num_classes*3*img_size*img_size [dtype = float32])
    N = (int) total number of samples in a batch
    OUTPUT = list of numpy array of same shape as input with augmented images
    """
    num_classes_in_batch = 0
    for imgs in list_imgs:
        if imgs is not None:
            num_classes_in_batch += 1

    imgs_per_class = int(math.ceil(N / num_classes_in_batch))
    out_list = []
    for imgs in list_imgs:
        if imgs is not None:
            if imgs.shape[0] >= imgs_per_class:
                np.random.shuffle(imgs)
                imgs = imgs[:imgs_per_class, :, :, :]
                out_list.append(imgs)
            else:
                imgs_in_cls = int(imgs.shape[0])
                num_augs = int(math.ceil(imgs_per_class / imgs_in_cls)) - 1
                imgs_to_append = imgs
                for i in range(num_augs):
                    np.random.shuffle(imgs)
                    aug_imgs = augment(imgs.transpose(0, 2, 3, 1))
                    imgs_to_append = np.vstack((imgs_to_append, aug_imgs.transpose(0, 3, 1, 2)))
                imgs_to_append = imgs_to_append[:imgs_per_class, :, :, :]
                out_list.append(imgs_to_append)
        else:
            out_list.append(None)

    return out_list


def Augment(img_batch, mask_batch):
    img_batch = img_batch
    mask_batch = mask_batch
    img_list = []
    img_s1_list = []
    # img_s2_list = []

    mask_list = []
    for i in range(img_batch.shape[0]):
        img = img_batch[i].squeeze(0)
        img = img.numpy()
        mask = mask_batch[i].numpy()
        # if random.random() > 0.5:
        #     img, mask = random_rot_flip(img, mask)
        # elif random.random() < 0.5:
        #     img, mask = random_rotate(img, mask)
        x, y = img.shape
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s1 = deepcopy(img)
        # img_s2 = deepcopy(img)

        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0
        # if random.random() < 0.8:
        #     img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        # img_s2 = blur(img_s2, p=0.5)
        # img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).long()

        img_list.append(img)
        img_s1_list.append(img_s1)
        # img_s2_list.append(img_s2)

        mask_list.append(mask)
    img = torch.stack(img_list)
    img_s1 = torch.stack(img_s1_list)
    # img_s2 = torch.stack(img_s2_list)

    mask = torch.stack(mask_list)
    # return img,img_s1,img_s2,mask

    return img, img_s1, mask


def Augment_unl(img_batch):
    img_batch = img_batch
    img_list = []
    img_s1_list = []
    # img_s2_list = []

    mask_list = []
    for i in range(img_batch.shape[0]):
        img = img_batch[i].squeeze(0)
        img = img.numpy()
        x, y = img.shape
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s1 = deepcopy(img)
        # img_s2 = deepcopy(img)

        img = torch.from_numpy(np.array(img)).unsqueeze(0).float() / 255.0
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0
        # if random.random() < 0.8:
        #     img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        # img_s2 = blur(img_s2, p=0.5)
        # img_s2 = torch.from_numpy(np.array(img_s2)).unsqueeze(0).float() / 255.0

        img_list.append(img)
        img_s1_list.append(img_s1)
        # img_s2_list.append(img_s2)

    img = torch.stack(img_list)
    img_s1 = torch.stack(img_s1_list)
    # img_s2 = torch.stack(img_s2_list)

    return img, img_s1


def Augment_one(img_batch):
    img_batch = img_batch
    img_s1_list = []
    for i in range(img_batch.shape[0]):
        img = img_batch[i].squeeze(0)
        img = img.numpy()
        img = Image.fromarray((img * 255).astype(np.uint8))
        img_s1 = deepcopy(img)
        if random.random() < 0.5:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        if random.random() < 0.5:
            img_s1 = blur(img_s1, p=0.5)
        img_s1 = torch.from_numpy(np.array(img_s1)).unsqueeze(0).float() / 255.0
        img_s1_list.append(img_s1)
    img_s1 = torch.stack(img_s1_list)
    return img_s1


# aug_GaussianNoise = Compose([  # RandomRotate90((1, 2), p=0.5)
#     # Flip(0, p=0.5),Flip(2, p=0.5)
#     # Rotate((-15, 15), (0, 0), (0, 0), p=0.5),
#     # RandomCropFromBorders(crop_value=0.1, p=0.5),
#     # ElasticTransform((0, 0.25), interpolation=2, p=0.1),
#     # RandomDropPlane(plane_drop_prob=0.1, axes=(0, 1, 2), p=0.5),
#     # RandomCrop(patch_size)
#     GaussianNoise(var_limit=(0, 5), p=0.5)
#
#     # RandomGamma(gamma_limit=(0.5, 1.5), p=0.5)
# ], p=1.0)


def Augment_3D(img_batch, mask_batch):
    img_list = []
    img_s1_list = []

    mask_list = []
    for i in range(img_batch.shape[0]):
        img = img_batch[i].squeeze(0)
        img = img.numpy()
        mask = mask_batch[i].numpy()
        # if random.random() > 0.5:
        #     img, mask = random_rot_flip(img, mask)
        # elif random.random() < 0.5:
        #     img, mask = random_rotate(img, mask)
        img = (img * 255).astype(np.uint8)
        img_s1 = deepcopy(img)

        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0

        if random.random() < 0.8:
            img_s1 = random_gamma(img_s1)
        img_s1 = random_blur(img_s1)
        img_s1 = torch.from_numpy(img_s1).unsqueeze(0).float() / 255.0

        mask = torch.from_numpy(np.array(mask)).long()

        img_list.append(img)
        img_s1_list.append(img_s1)

        mask_list.append(mask)
    img = torch.stack(img_list)
    img_s1 = torch.stack(img_s1_list)

    mask = torch.stack(mask_list)
    return img, img_s1, mask