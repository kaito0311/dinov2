import json
import numpy as np
import cv2
import os
from scipy.io import loadmat
import pickle
import time
import datetime
import logging
from PIL import Image, ImageFilter
from albumentations import center_crop



class Image_variation_gen:
    @staticmethod
    def circle_image(image):
        '''
        mask image with circle 
        '''
        mask = np.zeros_like(image)
        mask = cv2.circle(
            mask, (image.shape[0] // 2, image.shape[1] // 2), image.shape[0] // 2, (1, 1, 1), -1)

        return image * mask

    @staticmethod
    def degrade_image_to_numpy_array(image, quality, downsize, upsize=224, sigma=0.5):
        """Degrades an image to a NumPy array by JPEG compression, resizing, and adding a Gaussian blur.

        Args:
        image: A PIL Image object.
        quality: The JPEG compression quality (0-100).
        scale: The scale factor to resize the image by.
        sigma: The standard deviation of the Gaussian blur.

        Returns:
        A NumPy array representing the degraded image.
        """

        # Resize the image.
        image = Image.fromarray(image)
        resized_image = image.resize((int(downsize), int(downsize)))

        # Add a Gaussian blur.
        blurred_image = resized_image.filter(ImageFilter.GaussianBlur(sigma))

        # Convert the degraded image to a JPEG buffer.
        blurred_image = np.array(blurred_image)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode('.jpg', blurred_image, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        decimg = cv2.resize(decimg, (upsize, upsize), cv2.INTER_CUBIC)
        decimg = Image_variation_gen.circle_image(decimg)

        return decimg

    @staticmethod
    def down_up(image, down_size: int = 24, up_size: int = 224):
        '''
        Down scale and up scale image 
        '''
        # h, w, _ = image.shape
        # while h//2 > down_size and w // 2 > down_size:
        #     image = cv2.resize(image, (w//2, h//2), interpolation= cv2.INTER_CUBIC)
        #     h, w, _ = image.shape

        # image = cv2.resize(image, (down_size, down_size), interpolation = cv2.INTER_LINEAR)

        # image = cv2.resize(image, (up_size, up_size), interpolation= cv2.INTER_LINEAR)

        image = Image.fromarray(np.copy(np.array(image)))
        image = image.resize((down_size, down_size), Image.BICUBIC)
        image = image.resize((up_size, up_size), Image.BICUBIC)
        image = np.copy(np.array(image))

        return image
        return cv2.resize(cv2.resize(image, (down_size, down_size)), (up_size, up_size))

    @staticmethod
    def variation_image_to_embed(image: np.ndarray, ratio_stride=0.35, ls_down_size=[24, 42, 64], up_size=224, save_original=False, ls_variation_processed=[]) -> list:
        '''
        stride square through image to create square variation of original image 

        Arguments: 
            image: (H, W, 3)
            ratio_stride: stride  = 0.1 * min(H, W) of image 

        Returns: 
            list variation of original image 
        '''

        h, w, _ = image.shape
        size_variation = min(h, w)
        ls_variation = []
        ls_type_variation = []
        ls_variation_processed = [str(i) for i in ls_variation_processed]
        # center crop
        image_crop = center_crop(image, crop_height=min(
            image.shape[0], image.shape[1]), crop_width=min(image.shape[0], image.shape[1]))
        for downsize in ls_down_size:
            name_variation = str(downsize)
            if name_variation in ls_variation_processed:
                continue
            image_crop_out = Image_variation_gen.down_up(
                image_crop, down_size=downsize, up_size=up_size)
            ls_variation.append(
                Image_variation_gen.circle_image(image_crop_out))
            ls_type_variation.append(name_variation)

        if save_original:
            name_variation = str("original")
            if name_variation not in ls_variation_processed:
                ls_variation.append(cv2.resize(
                    image, (224, 224), interpolation=cv2.INTER_CUBIC))
                ls_type_variation.append(name_variation)

        # variation
        stride = int(min(h, w) * ratio_stride)
        if not (stride > 0):
            return ls_variation, ls_type_variation
        
        if stride < min(ls_down_size): return ls_variation, ls_type_variation

        for index_height in range(0, h, stride):
            if index_height + size_variation > h:
                break
            for index_weight in range(0, w, stride):
                if index_weight + size_variation > w:
                    break
                shortcut_temp = (image[index_height: index_height + size_variation,
                                 index_weight: index_weight + size_variation, :])
                if shortcut_temp.shape[0] == shortcut_temp.shape[1]:
                    # ls_variation.append(circle_image(tmp))

                    for down_size in ls_down_size:
                        tmp = np.copy(shortcut_temp)
                        name_variation = str(
                            index_height) + "_" + str(index_weight) + "_" + str(down_size)
                        if name_variation in ls_variation_processed:
                            continue
                        variation_temp = Image_variation_gen.down_up(
                            tmp, down_size=down_size, up_size=up_size)
                        variation_temp = Image_variation_gen.circle_image(
                            variation_temp)
                        ls_variation.append(np.copy(variation_temp))
                        ls_type_variation.append(name_variation)

        for index_height in range(-h, 0, stride):
            if index_height + size_variation > 0:
                break
            index_height = -index_height
            for index_weight in range(-w, 0, stride):
                if index_weight + size_variation > 0:
                    break
                index_weight = -index_weight
                shortcut_temp = (image[index_height - size_variation: index_height,
                                 index_weight - size_variation: index_weight, :])
                if shortcut_temp.shape[0] == shortcut_temp.shape[1] and shortcut_temp.shape[0] > 0:
                    for down_size in ls_down_size:
                        tmp = np.copy(shortcut_temp)
                        name_variation = str(
                            index_height) + "_" + str(index_weight) + "_" + str(down_size)
                        if name_variation in ls_variation_processed:
                            continue
                        variation_temp = Image_variation_gen.down_up(
                            tmp, down_size=down_size, up_size=up_size)
                        variation_temp = Image_variation_gen.circle_image(
                            variation_temp)
                        ls_variation.append(np.copy(variation_temp))
                        ls_type_variation.append(name_variation)

                break
            break

        assert len(ls_variation) == len(ls_type_variation)

        return ls_variation, ls_type_variation

