# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import random 

import cv2
import numpy as np 
from PIL import Image 
from torchvision import transforms
from albumentations import center_crop

from dinov2.data.avatar_augment.transforms import (
    transform_resize, transform_JPEGcompression, 
    transform_adjust_gamma, transform_to_gray, transform_color_jiter
)
from .transforms import (
    GaussianBlur,
    make_normalize_transform,
)
from dinov2.data.avatar_augment.variation import Image_variation_gen


logger = logging.getLogger("dinov2")


class DataAugmentationDINO(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(image)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(image)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output

class DataAugmentationAvatarSearch(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        if local_crops_number != 2: 
            raise ValueError("local_crops_number current must be 2")
    

        # normalization
        self.normalize_global = transforms.Compose(
            [
                transforms.Resize((global_crops_size, global_crops_size), interpolation= transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )
        self.normalize_local = transforms.Compose(
            [
                transforms.Resize((global_crops_size, global_crops_size), interpolation= transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                make_normalize_transform(),
            ]
        )

    def augment_pipeline(self, image: Image, padding_to_keep_ratio= False): 


        # Downscale augmentation
        if np.random.uniform() < 0.8:
            image = transform_resize(image, resize_range = (20, 64))

        # Gamma augmentation
        if np.random.uniform() < 0.3:
            image = transform_adjust_gamma(image, 0.8, 1.1)

        # JPEG augmentation
        if np.random.uniform() < 0.5:
            image = transform_JPEGcompression(image, compress_range = (10, 80))
        
        if np.random.uniform() < 0.3: 
            image = transform_color_jiter(image, brightness= 0.1, contrast= 0.1, saturation= 0.1, hue = 0.03)
        

        if padding_to_keep_ratio:
            image, mask = self.add_padding_image(image) 
        
        # if np.random.rand() < 1.0 and is_random_crop:
        #     w, h = image.size 
        #     if min(w, h) >= 224:
        #         random_crop = albumentations.augmentations.crops.transforms.RandomResizedCrop(224, 224, scale=(0.7,1), ratio=(1,1), always_apply= True)
        #         image = np.array(image) 
        #         image = random_crop(image=image)["image"]
        #         image = Image.fromarray(image)
        

        # Circle image 
        image = np.array(image)
        mask = np.zeros_like(image)
        mask = cv2.circle(
            mask, (image.shape[0] // 2, image.shape[1] // 2), image.shape[0] // 2, (1, 1, 1), -1)
        image = Image.fromarray(image * mask)
        return image

    def augment_db_avatarsearch(self, image: Image):

        list_variation, list_type_variation = Image_variation_gen.variation_image_to_embed(
            np.array(image), 0.35, ls_down_size=[20, 32, 54],
            save_original=True
        )
        idx = random.choice(range(len(list_variation))) 

        return list_variation[idx], list_type_variation[idx]

    def random_crop(self, image, start_row, start_col): 
        image = np.array(image)
        w, h, _ = image.shape 
        
        if w < h: 
            if start_col + w > h: 
                start_col = h - w 
            
            delta = int(w * 0.35 / 2)
            start_col = random.randint(-delta, delta) + start_col 
            
            if start_col + w > h:
                start_col = h - w
            
            if start_col < 0: start_col = 0

            image_crop = image[:, start_col: start_col + w, :]
        elif w > h:
            if start_row + h > w:
                start_row = w - h 

            delta = int(h * 0.35 / 2)
            start_row = random.randint(-delta, delta) + start_row

            if start_row + h > w: 
                start_row = w - h
            
            if start_row < 0: start_row = 0
            
            image_crop = image[start_row:start_row + h, :, :]
        else:
            image_crop = image 

        return image_crop 

    def get_a_pair(self, image):
        image_2, type_variation = self.augment_db_avatarsearch(image.copy())
        image_2 = Image.fromarray(image_2)
        if "_" not in type_variation: 
            if type_variation != "original":
                image = np.array(image)
                image_center_crop = center_crop(image, crop_height=min(
                    image.shape[0], image.shape[1]), crop_width=min(image.shape[0], image.shape[1]))
                
                image_center_crop = Image.fromarray(image_center_crop)
                image_1 = self.augment_pipeline(image_center_crop)
            else:
                image_1 = self.augment_pipeline(image.resize((224,224))) 

        else:
            start_row, start_col, _ = type_variation.split("_")
            start_row, start_col = int(start_row), int(start_col)
            image_1 = self.random_crop(image, start_row, start_col)
            image_1 = Image.fromarray(image_1)
            image_1 = self.augment_pipeline(image_1)
        
        
        return self.normalize_global(image_1), self.normalize_local(image_2) 


    def __call__(self, image):
        output = {}

        global_crop_1, local_crop_1 = self.get_a_pair(image.copy())
        global_crop_2, local_crop_2 = self.get_a_pair(image.copy()) 


        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            local_crop_1, local_crop_2
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        return output
