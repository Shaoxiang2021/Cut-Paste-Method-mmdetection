import os
import cv2
import numpy as np
import math
from tqdm import tqdm
from path import ROOT_DIR

def get_output_paths(ctr:int, n_images:int, out_path):

    if ctr <= int(n_images*0.8):  # train
        img_out_path = os.path.join(out_path, 'train', 'imgs')
        mask_out_path = os.path.join(out_path, 'train', 'masks')

    else:
        img_out_path = os.path.join(out_path, 'val', 'imgs')
        mask_out_path = os.path.join(out_path, 'val', 'masks')

    return img_out_path, mask_out_path

def create_directories(out_path):

    sub_dir = ['train', 'val']
    for dir in sub_dir:
        # set absolute paths
        img_out_path = os.path.join(out_path, dir, 'imgs')
        mask_out_path = os.path.join(out_path, dir, 'masks')
        # create directories
        os.makedirs(img_out_path, exist_ok=True)
        os.makedirs(mask_out_path, exist_ok=True)
        # write gitignore file to exclude synthetic img data from tracking

def get_templates(objs, obj_path, train_resolution):

    templates = [] # empty 2d list for storing all templates (row: template_ident(e.g. metal), coloumn: corresponding template objs)

    for obj in objs:
        print("load cut images for {} ...".format(obj))
        obj_list = [] # temporary list

        folder_path = os.path.join(obj_path, obj)
        file_list = os.listdir(folder_path)
        for file_name in tqdm(file_list, total=len(file_list)):
            # test if file is an .png image
            if not os.fsdecode(file_name).endswith(".png"):
                continue
            # read image and store in respective vector
            
            img = cv2.cvtColor(cv2.imread(os.path.join(obj_path, obj, file_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2RGBA)/255
            
            ratio = img.shape[0]/img.shape[1]
            obj_list.append(cv2.resize(img, (train_resolution[0], math.ceil(train_resolution[0]*ratio)), cv2.INTER_LINEAR))
            
        templates.append(obj_list)
    
    return templates

def load_canvas(canvas_folderpath, size_x, size_y):

    # set directory
    #canvas_folderpath = os.path.join(ROOT_DIR,'canvas')     
    canvas_fileName = np.random.choice(os.listdir(canvas_folderpath)) #select rand canvas from pool
    canvas_filePath = os.path.join(canvas_folderpath, canvas_fileName)
    
    # read canvas and convert it to RGBA
    canvas = cv2.cvtColor(cv2.imread(canvas_filePath, cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2RGBA)/255

    canvas = cv2.resize(canvas.copy(), (size_x, size_y), interpolation = cv2.INTER_NEAREST)
    canvas[:, :, 3] = 0
    canvas = np.dstack((canvas, canvas[:, :, 3]))

    return(canvas)
    
def generate_config_for_training(config_name, folder_name):

    config_path = os.path.join(ROOT_DIR, 'mmdetection', 'configs', 'romafo', config_name)
    
    with open(config_path, 'r') as file:
        config_content = file.readlines()
    
    data_root = os.path.join('..', 'data', 'synthetic_images', folder_name)
    
    work_dir = os.path.join('..', 'results', config_name.split('_')[0] + '_' + folder_name)
    
    for i, line in enumerate(config_content):
        if 'DATA_ROOT' in line:
            config_content[i] = f"DATA_ROOT = '{data_root}{os.sep}'\n"
        elif 'work_dir' in line:
            config_content[i] = f"work_dir = '{work_dir}'\n"
        
        # only write first 15 line, here will be improved
        if i >= 15:
            break  
      
    with open(config_path, 'w') as file:
        file.writelines(config_content)
