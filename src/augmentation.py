import cv2
import numpy as np
import math
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import skimage.transform as transform
from handling import get_templates
from path import ROOT_DIR
import os

class AugmentationGenerator(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.source_root = os.path.join(ROOT_DIR, 'data', 'source_images')

        self.height_ratio = self.train_resolution[1]/self.camera_resolution[1]
        self.width_ratio = self.train_resolution[0]/self.camera_resolution[0]

        self.load_source_images()
        self.initialize_objects_info()     

        # initialize transform for data augmentation
        self.initialize_transform()

    ### for initialization
    def calculate_objects_info(self, src_list, src_dic):
        
        src_info = dict()
        for cls in src_list:
            src_info[cls] = (math.ceil(src_dic[cls][0]*self.width_ratio), math.ceil(src_dic[cls][1]*self.height_ratio))
        
        return src_info

    def initialize_objects_info(self):
        
        if self.resize_in_procent is False:
            self.obj_info = self.calculate_objects_info(self.obj_list, self.obj_dic)
            if self.load_dist is True:
                self.dist_info = self.calculate_objects_info(self.dist_list, self.dist_dic)
    
    def load_source_images(self):
        
        print("initialized and loading cut images ...")
        self.src_imgs = get_templates(self.obj_list, os.path.join(self.source_root, '04_crop'), self.train_resolution)

        if self.load_dist is True:
            self.dist_imgs = get_templates(self.dist_list, os.path.join(self.source_root, '06_distractor'), self.train_resolution)
    
    def initialize_transform(self):

        # initialize transform for changing geometry
        #if self.scale_strategy == 'NORMAL' or 'MIX':
            # changing geometry
        #    self.transform_geometry = v2.Compose([
        #                            v2.ToImage(),
        #                            v2.RandomRotation(degrees=self.max_degrees, interpolation=InterpolationMode.BILINEAR, fill=0),
        #                            v2.RandomPerspective(distortion_scale=self.distortion_scale, p=self.general_probability, interpolation=InterpolationMode.BILINEAR, fill=0),
        #                            v2.RandomHorizontalFlip(p=self.general_probability),
        #                           v2.RandomVerticalFlip(p=self.general_probability),
        #                            ])
        
        #elif self.scale_strategy == 'ORIGINAL_MIX':
        #    self.transform_geometry = v2.Compose([
        #                            v2.ToImage(),
        #                            v2.RandomRotation(degrees=self.max_degrees, interpolation=InterpolationMode.BILINEAR, fill=0),
        #                            v2.RandomPerspective(distortion_scale=self.distortion_scale, p=self.general_probability, interpolation=InterpolationMode.BILINEAR, fill=0),
        #                            v2.RandomHorizontalFlip(p=self.general_probability),
        #                            v2.RandomVerticalFlip(p=self.general_probability),
        #                            v2.RandomAffine(degrees=0, scale=(self.min_scale_factor, self.max_scale_factor), interpolation=InterpolationMode.BILINEAR, fill=0),
        #                            ])
        
        # !!! ORIGINAL_MIX is not used anymore !!!
        
        self.transform_geometry = v2.Compose([
                                    v2.ToImage(),
                                    v2.RandomRotation(degrees=self.max_degrees, interpolation=InterpolationMode.BILINEAR, fill=0),
                                    v2.RandomPerspective(distortion_scale=self.distortion_scale, p=self.general_probability, interpolation=InterpolationMode.BILINEAR, fill=0),
                                    v2.RandomHorizontalFlip(p=self.general_probability),
                                    v2.RandomVerticalFlip(p=self.general_probability),
                                    ])
        
        # initialize transform for adding image effects
        self.transform_image_effect = v2.Compose([
                                        v2.ColorJitter(**self.color_jitter),
                                        #v2.GaussianBlur(**self.gaussian_blur),
                                    ])

    ### for main function
    def add_shadow(self, imgcan, mask):

        # add shadow
        shadow_dir = np.random.uniform(0, 2*np.pi) # shadow direction
        shadow_len = np.random.normal(0.8, 0.8) # shadow length (amount of translation)
        shadow_mask_trans = transform.AffineTransform(scale = 1.02, translation =(shadow_len * np.sin(shadow_dir), shadow_len * np.cos(shadow_dir)), shear=None)
        shadow_mask = transform.warp(mask.copy(), shadow_mask_trans.inverse, output_shape=(mask.shape[0], mask.shape[1]))
        shadow_mask = cv2.GaussianBlur(shadow_mask.copy(), (31, 31), 0)
        # apply to all color channels
        for i in range(3):
            imgcan[:,:,i] = np.subtract(imgcan[:,:,i].copy(), shadow_mask * self.shadow_strength)
        imgcan[:,:,:3] = np.clip(imgcan[:,:,:3].copy(), 0, 255)
        
        return imgcan
    
    def resize_object(self, img, obj_choice, scale, dist):

        contours, _ = cv2.findContours(img[:, :, 3].copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(np.vstack(contours))

        angle = rect[2]
        width = int(rect[1][0])
        height = int(rect[1][1])

        rotation_matrix = cv2.getRotationMatrix2D(tuple(rect[0]), angle, 1)
        rotated_image = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        cropped_image = rotated_image[int(rect[0][1]-height/2)-self.correct_factor:int(rect[0][1]+height/2)+self.correct_factor, int(rect[0][0]-width/2)-self.correct_factor:int(rect[0][0]+width/2)+self.correct_factor]
        if cropped_image.shape[0] > cropped_image.shape[1]:
            cropped_image = cv2.transpose(cropped_image)
            cropped_image = cv2.flip(cropped_image, 1)
        
        if dist is False:
            src_dic = self.obj_dic
            src_info = self.obj_info
        else:
            src_dic = self.dist_dic
            src_info = self.dist_info

        if self.keep_ratio is True and self.resize_in_procent is False:
            
            obj_width = math.ceil(src_info[list(src_dic.keys())[obj_choice]][0]*scale)
            ratio = cropped_image.shape[0]/cropped_image.shape[1]
            obj_height = math.ceil(obj_width*ratio)
            
            # print([width, height], [obj_width, obj_height], ratio, cropped_image.shape)
            cropped_image = cv2.resize(cropped_image, [obj_width, obj_height], cv2.INTER_LINEAR)

        elif self.keep_ratio is True and self.resize_in_procent is True:
            obj_width = math.ceil(src_dic[list(src_dic.keys())[obj_choice]]*self.train_resolution[0]*scale)
            ratio = cropped_image.shape[0]/cropped_image.shape[1]
            obj_height = math.ceil(obj_width*ratio)
            cropped_image = cv2.resize(cropped_image, [obj_width, obj_height], cv2.INTER_LINEAR)

        else: # not used anymore
            obj_size = [math.ceil(ele*scale) for ele in src_info[list(src_dic.keys())[obj_choice]]]
            cropped_image = cv2.resize(cropped_image, obj_size, cv2.INTER_LINEAR)

        return cropped_image
    
    def pre_roi(self, img, obj_choice, scale, dist):

        #if self.scale_strategy == 'NORMAL' or self.scale_strategy == 'MIX':
            
        #    obj = self.resize_object(img, obj_choice, scale, dist)
            
            # bug of cutting a part of object
            # max_border = math.ceil(np.max(obj.shape)*self.max_scale_factor + 2*self.correct_factor)
            
            # bug fixed
        #    max_border = math.ceil(np.sqrt(np.square(obj.shape[0])+np.square(obj.shape[1])))
        #    object_background = np.zeros((max_border, max_border, 4))

        #    x_place = math.ceil(max_border/2 - obj.shape[1]/2)
        #    y_place = math.ceil(max_border/2 - obj.shape[0]/2)

        #    object_background[y_place:y_place+obj.shape[0], x_place:x_place+obj.shape[1]] = obj

        #elif self.scale_strategy == 'ORIGINAL_MIX':
            # in this strategy there is no resize process

        #    cnts, _ = cv2.findContours((img[:, :, 3].copy()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #    x, y, w, h = cv2.boundingRect(np.vstack(cnts))
        #    obj = img[y-self.correct_factor:y+h+self.correct_factor, x-self.correct_factor:x+w+self.correct_factor]

        #    diagonal = math.ceil(np.sqrt(np.square(obj.shape[0])+np.square(obj.shape[1]))*self.max_scale_factor)
        #    object_background = np.zeros((diagonal, diagonal, 4))

        #    x_place = math.ceil(diagonal/2 - obj.shape[1]/2)
        #    y_place = math.ceil(diagonal/2 - obj.shape[0]/2)

        #    object_background[y_place:y_place+obj.shape[0], x_place:x_place+obj.shape[1]] = obj
        
        # !!! ORIGINAL_MIX is not used anymore !!!
        
        obj = self.resize_object(img, obj_choice, scale, dist)
    
        max_border = math.ceil(np.sqrt(np.square(obj.shape[0])+np.square(obj.shape[1])))
        object_background = np.zeros((max_border, max_border, 4))

        x_place = math.ceil(max_border/2 - obj.shape[1]/2)
        y_place = math.ceil(max_border/2 - obj.shape[0]/2)

        object_background[y_place:y_place+obj.shape[0], x_place:x_place+obj.shape[1]] = obj

        return object_background

    def img_aug(self, obj_choice, scale, dist=False):

        # select random image from pool
        if dist is False:
            #img = self.src_imgs[obj_choice][np.random.randint(0, len(self.src_imgs[obj_choice]))]
            obj_id = np.random.randint(0, len(self.src_imgs[obj_choice]))
            img = self.src_imgs[obj_choice][obj_id]
        else:
            img = self.dist_imgs[obj_choice][np.random.randint(0, len(self.dist_imgs[obj_choice]))] 

        # fill the small hole? maybe later

        # prepared roi
        # print(obj_choice, obj_id)
        object_background = self.pre_roi(img, obj_choice, scale, dist)

        # transform geometry 
        img_transformed = self.transform_geometry(object_background)
        img_transformed_rgb = img_transformed[:3, :, :]

        # apply color jitter
        img_transformed_rgb = self.transform_image_effect(img_transformed_rgb)
        img_transformed[:3, :, :] = img_transformed_rgb

        # convert to numpy array
        transform_PIL = v2.ToPILImage()
        img_transformed = np.array(transform_PIL(img_transformed))

        # crop the object and add in background in random position
        cnts, _ = cv2.findContours((img_transformed[:, :, 3].copy()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(np.vstack(cnts))
        # img_obj = img_transformed[y-self.correct_factor:y+h+self.correct_factor, x-self.correct_factor:x+w+self.correct_factor]/255

        # img_obj bug mask is zero, bug fixed
        img_obj = img_transformed[y:y+h, x:x+w]/255

        # add overlay channel
        img_obj = np.dstack((img_obj, img_obj[:, :, 3] > 0))
        
        return img_obj

    def imgcan_aug(self):
        pass
