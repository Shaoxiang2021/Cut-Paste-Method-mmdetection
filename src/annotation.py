from pycocotools import mask as maskUtils
from tqdm import tqdm
import numpy as np
import json
import copy
import os

class AnnotationGenerator(object):

    def __init__(self):

        self.coco_annotation_format = {
            'info': list(),
            'categories': list(),
            'images': list(),
            'annotations': list()
        }

        self.categories_format = {
            'id': int(),
            'name': str(),
            'supercategory': 'industry object'
        }

        self.images_format = {
            "id": int(),
            "width": int(),
            "height": int(),
            "file_name": str(),
            "license": 1,
            "flickr_url": str(),
            "coco_url": str(),
            "date_captured": str()
        }

        self.annotations_format = {
            "id": int(),
            "image_id": int(),
            "category_id": int(),
            "segmentation": dict(),
            "area": float(),
            "bbox": list(),
            "iscrowd": 0
        }

        self.info = list()
        self.categories = list()
        self.images = dict()
        self.annotations_list = list()

    def generate_info(self, info=list()):
        self.info = info

    def generate_categories(self, obj_list):

        for id in range(1, len(obj_list)+1):
            
            categories = copy.deepcopy(self.categories_format)
            categories['id'] = id
            categories['name'] = obj_list[id-1]
            self.categories.append(categories)

    def generate_images(self, id, width, height, file_name):

        self.images = copy.deepcopy(self.images_format)
        self.images['id'] = id
        self.images['width'] = width
        self.images['height'] = height
        self.images['file_name'] = file_name

    def generate_annotation(self, id, image_id, categories_id, mask):

        rle = maskUtils.encode(np.asfortranarray(mask))
        area = np.float64(maskUtils.area(rle))
        bbox = list(maskUtils.toBbox(rle))
        rle['counts'] = rle['counts'].decode('utf-8')

        annotation = copy.deepcopy(self.annotations_format)
        annotation['id'] = id
        annotation['image_id'] = image_id
        annotation['category_id'] = categories_id
        annotation['segmentation'] = rle
        annotation['area'] = area
        annotation['bbox'] = bbox
        
        self.annotations_list.append(annotation)

    def generate_annotations(self, mask_list):

        for ele in mask_list:
            self.generate_annotation(*ele)

    def generate_annotation_for_single_images(self, output_path):

        annotation_json_single_image = {
            'images': self.images,
            'annotations': self.annotations_list
        }

        with open(output_path, 'w') as json_file:
            json.dump(annotation_json_single_image, json_file, indent=4)

        # reset annotation_list
        self.annotations_list = list()
        self.images = dict()

    def post_processing(self, obj_list, output_root):

        sub_dir = ['train', 'val']
        # sub_dir = ['train', 'val', 'test']

        self.generate_categories(obj_list)

        for dir in sub_dir:
            folder_path = os.path.join(output_root, dir, "imgs")
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

            coco_annotation = copy.deepcopy(self.coco_annotation_format)
            coco_annotation['categories'] = self.categories

            for filename in tqdm(json_files, desc=f"processing {dir} ..."):
                json_path = os.path.join(folder_path, filename)
                with open(json_path, 'r') as json_file:
                    file_content = json.load(json_file)
                    coco_annotation['images'].append(file_content['images'])
                    for ann in file_content['annotations']:
                        coco_annotation['annotations'].append(ann)

            json_output_path = os.path.join(output_root, dir, "annotations.json")
            with open(json_output_path, 'w') as json_file:
                json.dump(coco_annotation, json_file, indent=4)    
