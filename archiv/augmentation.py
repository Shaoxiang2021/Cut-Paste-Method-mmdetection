import cv2
import numpy as np
import math
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import skimage.transform as transform

class AugmentationGenerator(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.height_ratio = self.train_resolution[1]/self.camera_resolution[1]
        self.width_ratio = self.train_resolution[0]/self.camera_resolution[0]
        
        if self.resize_in_procent is False:
            self.obj_info = dict()
            for cls in list(self.obj_dic.keys()):
                self.obj_info[cls] = (math.ceil(self.obj_dic[cls][0]*self.width_ratio), math.ceil(self.obj_dic[cls][1]*self.height_ratio))
        else:
            pass

        # initialize transform for data augmentation
        self.initialize_transform()
    
    def initialize_transform(self):

        # initialize transform for changing geometry
        if self.scale_strategy == 'NORMAL' or 'MIX':
            # changing geometry
            self.transform_geometry = v2.Compose([
                                    v2.ToImage(),
                                    v2.RandomRotation(degrees=self.max_degrees, interpolation=InterpolationMode.BILINEAR, fill=0),
                                    v2.RandomPerspective(distortion_scale=self.distortion_scale, p=self.general_probability, interpolation=InterpolationMode.BILINEAR, fill=0),
                                    v2.RandomHorizontalFlip(p=self.general_probability),
                                    v2.RandomVerticalFlip(p=self.general_probability),
                                    v2.ToPILImage()
                                    ])
        
        elif self.scale_strategy == 'ORIGINAL_MIX':
            self.transform_geometry = v2.Compose([
                                    v2.ToImage(),
                                    v2.RandomRotation(degrees=self.max_degrees, interpolation=InterpolationMode.BILINEAR, fill=0),
                                    v2.RandomPerspective(distortion_scale=self.distortion_scale, p=self.general_probability, interpolation=InterpolationMode.BILINEAR, fill=0),
                                    v2.RandomHorizontalFlip(p=self.general_probability),
                                    v2.RandomVerticalFlip(p=self.general_probability),
                                    v2.RandomAffine(degrees=0, scale=(self.min_scale_factor, self.max_scale_factor), interpolation=InterpolationMode.BILINEAR, fill=0),
                                    v2.ToPILImage()
                                    ])

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
    
    def resize_object(self, img, obj_choice, scale):

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

        if self.keep_ratio is True and self.resize_in_procent is False:
            
            obj_width = math.ceil(self.obj_info[list(self.obj_dic.keys())[obj_choice]][0]*scale)
            ratio = cropped_image.shape[0]/cropped_image.shape[1]
            obj_height = math.ceil(obj_width*ratio)
            cropped_image = cv2.resize(cropped_image, [obj_width, obj_height], cv2.INTER_LINEAR)

        elif self.keep_ratio is True and self.resize_in_procent is True:
            obj_width = math.ceil(self.obj_dic[list(self.obj_dic.keys())[obj_choice]]*self.train_resolution[0]*scale)
            ratio = cropped_image.shape[0]/cropped_image.shape[1]
            obj_height = math.ceil(obj_width*ratio)
            cropped_image = cv2.resize(cropped_image, [obj_width, obj_height], cv2.INTER_LINEAR)

        else:
            obj_size = [math.ceil(ele*scale) for ele in self.obj_info[list(self.obj_dic.keys())[obj_choice]]]
            cropped_image = cv2.resize(cropped_image, obj_size, cv2.INTER_LINEAR)

        return cropped_image
    
    def pre_roi(self, img, obj_choice, scale):

        if self.scale_strategy == 'NORMAL' or self.scale_strategy == 'MIX':
            
            obj = self.resize_object(img, obj_choice, scale)

            max_border = math.ceil(np.max(obj.shape)*self.max_scale_factor + 2*self.correct_factor)
            object_background = np.zeros((max_border, max_border, 4))

            x_place = math.ceil(max_border/2 - obj.shape[1]/2)
            y_place = math.ceil(max_border/2 - obj.shape[0]/2)

            object_background[y_place:y_place+obj.shape[0], x_place:x_place+obj.shape[1]] = obj

        elif self.scale_strategy == 'ORIGINAL_MIX':

            cnts, _ = cv2.findContours((img[:, :, 3].copy()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(np.vstack(cnts))
            obj = img[y-self.correct_factor:y+h+self.correct_factor, x-self.correct_factor:x+w+self.correct_factor]

            diagonal = math.ceil(np.sqrt(np.square(obj.shape[0])+np.square(obj.shape[1]))*self.max_scale_factor)
            object_background = np.zeros((diagonal, diagonal, 4))

            x_place = math.ceil(diagonal/2 - obj.shape[1]/2)
            y_place = math.ceil(diagonal/2 - obj.shape[0]/2)

            object_background[y_place:y_place+obj.shape[0], x_place:x_place+obj.shape[1]] = obj

        return object_background

    def img_aug(self, src_imgs, obj_choice, scale):

        # select random image from pool
        img = src_imgs[obj_choice][np.random.randint(0, len(src_imgs[obj_choice]))]

        # fill the small hole? maybe later

        # prepared roi
        object_background = self.pre_roi(img, obj_choice, scale)

        # transform geometry 
        img_transformed = np.array(self.transform_geometry(object_background))

        # crop the object and add in background in random position
        cnts, _ = cv2.findContours((img_transformed[:, :, 3].copy()).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(np.vstack(cnts))
        img_obj = img_transformed[y-self.correct_factor:y+h+self.correct_factor, x-self.correct_factor:x+w+self.correct_factor]/255

        # add overlay channel
        img_obj = np.dstack((img_obj, img_obj[:, :, 3] > 0))
        
        return img_obj

    def imgcan_aug(self):
        pass
