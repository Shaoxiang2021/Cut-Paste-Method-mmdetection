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

        if len(self.obj_list) == 1:
            self.colorboard = False
        else:
            self.colorboard = True

    def random_point(self):

        size_y, size_x = self.imgcan[:, :, 4].shape
        cut_mask = self.imgcan[:, :, 4][:size_y-self.obj.target_height, :size_x-self.obj.target_width].copy()
        zero_coords = np.column_stack(np.where(cut_mask == 0))
        
        if len(zero_coords) > 0:
            random_point = random.choice(zero_coords)
            return tuple(random_point)
        else:
            return None

    def check_overlay_condition(self, dist=False):

        self.overlay_mask[self.top_left_y:self.top_left_y+self.obj.target_height, self.top_left_x:self.top_left_x+self.obj.target_width] = self.obj.img[:, :, 4].copy()
        check_board = self.imgcan[:, :, 4].copy()[self.top_left_y:self.top_left_y+self.obj.target_height, self.top_left_x:self.top_left_x+self.obj.target_width]
        check_board += self.obj.img[:, :, 4].copy()

        random_overlay = random.random()
        if dist is True or random_overlay > self.overlay_factor:
            if check_board.max() < 2:
                return False
            else:
                return True
        else:
            if check_board.max() < 3:
                return False
            else:
                return True
        
    def add_object(self, obj_choice, num_obj, dist=False):
        
        obj_mask = self.obj.img[:, :, 3].copy()
        
        # add shadow
        obj_mask_backgroundsize = np.zeros((self.size_y, self.size_x))
        obj_mask_backgroundsize[self.top_left_y:self.top_left_y + self.obj.target_height, self.top_left_x:self.top_left_x + self.obj.target_width] = obj_mask
        self.imgcan[:, :, :3] = self.augmentationGenerator.add_shadow(self.imgcan[:, :, :3], obj_mask_backgroundsize)

        # add object
        roi = self.imgcan[self.top_left_y:self.top_left_y + self.obj.target_height, self.top_left_x:self.top_left_x + self.obj.target_width, :3].copy()
        # self.imgcan[self.top_left_y:self.top_left_y + self.obj.target_height, self.top_left_x:self.top_left_x + self.obj.target_width, :3] = (1 - obj_mask[:, :, None]) * roi + obj_mask[:, :, None] * self.obj.img[:, :, :3]
        self.no_blending(obj_mask, roi)
        self.imgcan[:, :, 4] += self.overlay_mask

        if dist is False:
            if self.colorboard is not True:
                self.imgcan[:, :, 3][self.overlay_mask != 0] = num_obj
            else:
                self.imgcan[:, :, 3][self.overlay_mask != 0] = obj_choice + 1
        else:
            pass

    def no_blending(self, obj_mask, roi):
        self.imgcan[self.top_left_y:self.top_left_y + self.obj.target_height, self.top_left_x:self.top_left_x + self.obj.target_width, :3] = (1 - obj_mask[:, :, None]) * roi + obj_mask[:, :, None] * self.obj.img[:, :, :3]

    def reflash_mask(self):
        
        check_board = self.imgcan[:, :, 4].copy()
        check_board[check_board > 0] = 1
        check_board += self.overlay_mask

        for i in range(len(self.mask_list)-1):
            self.mask_list[i][-1][np.where(check_board == 2)] = 0

    #def load_hook(self, num_img):
    #
    #    if num_img in self.hook.keys():
    #        paras = ParameterManager(**self.hook[num_img])
    #        self.min_obj = paras.min_obj
    #        self.max_obj = paras.max_obj
    #       self.overlay_factor = paras.overlay_factor
    #        self.use_dist = paras.add_distractors
    #        self.augmentationGenerator.use_dist = paras.add_distractors
    
    def load_hook(self, num_img):
        
        if num_img in self.hook.keys(): 
            for key, value in self.hook[num_img].items():
                setattr(self, key, value)

    def image_generation(self):
        
        print("create direcotories for the data generation ...")
        # create directories
        create_directories(self.output_root)

        # copy config file for different settings
        print("copy config files for the different settings ...")
        shutil.copy('config.py', self.output_root)
        
        # print("prepared and loading cut images ...")
        # load cuting images
        # src_imgs = get_templates(list(self.obj_dic.keys()), os.path.join(self.source_root, '04_crop'), self.augmentationGenerator.train_resolution)

        # set seed for torch random
        if self.seed:
            torch.manual_seed(self.seed)

        print("generate images and coco labels ...")

        # initialize segmentation id
        id_counter = 1

        # generate images
        for num_image in tqdm(range(1, self.num_images+1)):
        
            # hook
            self.load_hook(num_image)
            
            # load background image
            self.imgcan = load_canvas(os.path.join(self.source_root, '01_canvas'), self.size_x, self.size_y)
            
            scale = None
            # choose object class and scale for 'NORMAL' strategy for the generation
            if self.generation_strategy == 'NORMAL':
                obj_choice = np.random.randint(0, len(self.augmentationGenerator.src_imgs))

            if self.scale_strategy == 'NORMAL':
                scale = random.uniform(self.augmentationGenerator.min_scale_factor, self.augmentationGenerator.max_scale_factor)

            # initialize list for single object mask
            self.mask_list = list()

            ### add objects
            for num_obj in range(1, np.random.randint(self.min_obj+1, self.max_obj+1)):
                
                # choose object class and scale for 'MIX' strategy for the generation
                if self.generation_strategy == 'MIX':
                    obj_choice = np.random.randint(0, len(self.augmentationGenerator.src_imgs))

                if self.scale_strategy == 'MIX':
                    scale = random.uniform(self.augmentationGenerator.min_scale_factor, self.augmentationGenerator.max_scale_factor)

                # add augmentation to the cuting object
                self.obj.img = self.augmentationGenerator.img_aug(obj_choice, scale)
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
            
            ### add distractors
            if self.use_dist is True:
                for num_dist in range(1, np.random.randint(5, 12)):
                    obj_choice = np.random.randint(0, len(self.augmentationGenerator.dist_imgs))
                    scale = random.uniform(self.augmentationGenerator.min_scale_factor, self.augmentationGenerator.max_scale_factor)
                    self.obj.img = self.augmentationGenerator.img_aug(obj_choice, scale, dist=True)
                    self.obj.target_height, self.obj.target_width, _ = self.obj.img.shape

                    iter_add_obj = 0
                    while iter_add_obj <= self.max_iter:
                        iter_add_obj += 1
                        self.top_left_y, self.top_left_x = self.random_point()
                        self.overlay_mask = np.zeros(self.imgcan[:, :, 4].shape, dtype=np.uint8)
                        if self.check_overlay_condition(dist=True) is not True:
                            self.add_object(obj_choice, num_dist, dist=True)
                            break
                        else:
                            pass # without add object

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
    
        self.annotationGenerator.post_processing(self.obj_list, self.output_root)
          
        print("completed ...")
