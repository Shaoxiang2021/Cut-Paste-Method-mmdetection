import numpy as np
import random
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import shutil
import random

from handling import *
from augmentation import AugmentationGenerator
from annotation import AnnotationGenerator
from path import ROOT_DIR

class Sample(object):
    pass

class ParameterManager(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class ImageGenerator(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # initialization
        self.obj = Sample()
        self.annotationGenerator =AnnotationGenerator()
        self.augmentationGenerator = AugmentationGenerator(**self.aug_params)

        self.source_root = os.path.join(ROOT_DIR, 'data', 'source_images')
        self.output_root = os.path.join(ROOT_DIR, 'data', 'synthetic_images', self.folder_name)

        if len(self.obj_dic) == 1:
            self.colorboard = False
        else:
            self.colorboard = True

    def random_point(self):

        size_y, size_x = self.imgcan[:, :, 4].shape
        cut_mask = self.imgcan[:, :, 4][:size_y-self.obj.target_height, :size_x-self.obj.target_width]
        zero_coords = np.column_stack(np.where(cut_mask == 0))
        
        if len(zero_coords) > 0:
            random_point = random.choice(zero_coords)
            return tuple(random_point)
        else:
            return None

    def check_overlay_condition(self):

        self.overlay_mask[self.top_left_y:self.top_left_y+self.obj.target_height, self.top_left_x:self.top_left_x+self.obj.target_width] = self.obj.img[:, :, 4]
        check_board = self.imgcan[:, :, 4].copy()
        check_board += self.overlay_mask

        random_overlay = random.random()
        if random_overlay > self.overlay_factor:
            if check_board.max() < 2:
                return False
            else:
                return True
        else:
            if check_board.max() < 3:
                return False
            else:
                return True
        
    def add_object(self, obj_choice, num_obj):

        obj_mask = self.obj.img[:, :, 3]
        
        # add shadow
        obj_mask_backgroundsize = np.zeros((self.size_y, self.size_x))
        obj_mask_backgroundsize[self.top_left_y:self.top_left_y + self.obj.target_height, self.top_left_x:self.top_left_x + self.obj.target_width] = obj_mask
        self.imgcan[:, :, :3] = self.augmentationGenerator.add_shadow(self.imgcan[:, :, :3], obj_mask_backgroundsize)

        # add object
        roi = self.imgcan[self.top_left_y:self.top_left_y + self.obj.target_height, self.top_left_x:self.top_left_x + self.obj.target_width, :3]
        self.imgcan[self.top_left_y:self.top_left_y + self.obj.target_height, self.top_left_x:self.top_left_x + self.obj.target_width, :3] = (1 - obj_mask[:, :, None]) * roi + obj_mask[:, :, None] * self.obj.img[:, :, :3]
        self.imgcan[:, :, 4] += self.overlay_mask

        if self.colorboard is not True:
            self.imgcan[:, :, 3][self.overlay_mask != 0] = num_obj
        else:
            self.imgcan[:, :, 3][self.overlay_mask != 0] = self.obj_dic[list(self.obj_dic.keys())[obj_choice]]

    def reflash_mask(self):
        
        check_board = self.imgcan[:, :, 4].copy()
        check_board[check_board > 0] = 1
        check_board += self.overlay_mask

        for i in range(len(self.mask_list)-1):
            self.mask_list[i][-1][np.where(check_board == 2)] = 0

    def image_generation(self):
        
        # create directories
        create_directories(self.output_root)

        # copy config file for different settings
        shutil.copy('config.py', self.output_root)
        
        print("prepared and loading cut images ...")
        # load cuting images
        src_imgs = get_templates(list(self.obj_dic.keys()), os.path.join(self.source_root, '04_crop'), self.augmentationGenerator.train_resolution)

        # set seed for torch random
        if self.seed:
            torch.manual_seed(self.seed)

        print("generate images and coco labels ...")

        # initialize segmentation id
        id_counter = 1

        # generate images
        for num_image in tqdm(range(1, self.num_images+1)):
            
            # load background image
            self.imgcan = load_canvas(os.path.join(self.source_root, '01_canvas'), self.size_x, self.size_y)
            
            scale = None
            # choose object class and scale for 'NORMAL' strategy for the generation
            if self.generation_strategy == 'NORMAL':
                obj_choice = np.random.randint(0, len(src_imgs))

            if self.augmentationGenerator.scale_strategy == 'NORMAL':
                scale = random.uniform(self.augmentationGenerator.min_scale_factor, self.augmentationGenerator.max_scale_factor)

            # initialize list for single object mask
            self.mask_list = list()

            # generate objects
            for num_obj in range(1, np.random.randint(self.min_obj+1, self.max_obj+1)):
                
                # choose object class and scale for 'MIX' strategy for the generation
                if self.generation_strategy == 'MIX':
                    obj_choice = np.random.randint(0, len(src_imgs))

                if self.augmentationGenerator.scale_strategy == 'MIX':
                    scale = random.uniform(self.augmentationGenerator.min_scale_factor, self.augmentationGenerator.max_scale_factor)

                # add augmentation to the cuting object
                self.obj.img = self.augmentationGenerator.img_aug(src_imgs, obj_choice, scale)
                self.obj.target_height, self.obj.target_width, _ = self.obj.img.shape

                iter_add_obj = 0
                while iter_add_obj <= self.max_iter:
                    iter_add_obj += 1
                    
                    # select random point in map
                    self.top_left_y, self.top_left_x = self.random_point()

                    # initialize overlay mask
                    self.overlay_mask = np.zeros(self.imgcan[:, :, 4].shape, dtype=np.uint8)

                    # check overly condition, if False then add object in background image
                    if self.check_overlay_condition() is not True:

                        self.add_object(obj_choice, num_obj)

                        if self.annotation_strategy == 'SPECIAL':
                            # annotation for single segmentation
                            self.annotationGenerator.generate_annotation(id_counter, num_image, obj_choice+1, self.overlay_mask.copy())

                        else:
                            self.mask_list.append([id_counter, num_image, obj_choice+1, self.overlay_mask.copy()])
                            self.reflash_mask()

                        id_counter += 1

                        break

                    else:
                        pass # without add object

            # image augmentation maybe gaussian blur?

            img_out_path, mask_out_path = get_output_paths(num_image, self.num_images, self.output_root)

            if self.generate_mask is True:
                mask_path = os.path.join(mask_out_path, 'mask{}.png'.format(num_image))
                plt.imsave(mask_path, self.imgcan[:, :, 3])

            img_path = os.path.join(img_out_path, 'imgcan{}.png'.format(num_image))
            cv2.imwrite(img_path, (self.imgcan[:, :, :3]*255).astype(np.uint8))

            if self.annotation_strategy == 'NORMAL':
                self.annotationGenerator.generate_annotations(self.mask_list)

            # annotation for single image
            self.annotationGenerator.generate_images(num_image, self.size_x, self.size_y, 'imgcan{}.png'.format(num_image))
            self.annotationGenerator.generate_annotation_for_single_images(os.path.join(img_out_path, 'imgcan{}.json'.format(num_image)))
        
        # post processing for json files
        
        print("generate coco format json files ...")
    
        self.annotationGenerator.post_processing(self.obj_dic, self.output_root)
          
        print("completed ...")
