import os
import cv2
import numpy as np
import math
import random
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

def get_templates(objs, obj_path, train_resolution, num_source_image):

    templates = [] # empty 2d list for storing all templates (row: template_ident(e.g. metal), coloumn: corresponding template objs)

    for obj in objs:
        print("load cut images for {} ...".format(obj))
        obj_list = [] # temporary list

        folder_path = os.path.join(obj_path, obj)
        file_list = os.listdir(folder_path)
        
        len_list = len(file_list)
        if len_list >= num_source_image:
            len_list = num_source_image
        else:
            pass
        
        source_id = 0
        
        for file_name in tqdm(file_list, total=len_list):
            # test if file is an .png image
            if not os.fsdecode(file_name).endswith(".png"):
                continue
            # read image and store in respective vector
            
            if source_id >= num_source_image:
                break
            
            img = cv2.cvtColor(cv2.imread(os.path.join(obj_path, obj, file_name), cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2RGBA)/255
            ratio = img.shape[0]/img.shape[1]
            img = cv2.resize(img, (train_resolution[0], math.ceil(train_resolution[0]*ratio)), cv2.INTER_LINEAR)
            
            obj_list.append(img)
            
            source_id += 1
            
        templates.append(obj_list)
    
    return templates

def load_canvas(canvas_folderpath, size_x, size_y):

    # set directory
    #canvas_folderpath = os.path.join(ROOT_DIR,'canvas')     
    canvas_fileName = np.random.choice(os.listdir(canvas_folderpath)) #select rand canvas from pool
    canvas_filePath = os.path.join(canvas_folderpath, canvas_fileName)
    
    # read canvas and convert it to RGBA
    # bug fixed BGR2RGBA not RGB2RGBA
    canvas = cv2.cvtColor(cv2.imread(canvas_filePath, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA)/255

    canvas = cv2.resize(canvas, (size_x, size_y), interpolation = cv2.INTER_NEAREST)
    
    # canvas augmentation
    
    # random flip
    if np.random.rand() < 0.5:
        flip_code = np.random.randint(-1, 2)
        canvas = cv2.flip(canvas, flip_code)
    
    rgb_cavas = canvas[:, :, :3]
    alpha_cavas = canvas[:, :, 3]
    
    # random gaussian blur and brightness
    if np.random.rand() < 0.5:
        blur_amount = np.random.randint(1, 6)*2 + 1
        rgb_cavas = cv2.GaussianBlur(rgb_cavas, (blur_amount, blur_amount), 0)
        rgb_cavas = np.clip(cv2.multiply(rgb_cavas, random.uniform(0.5, 1.5)), 0, 1)
    
    mask_cavas = np.zeros_like(alpha_cavas)
    overlay_cavas = np.zeros_like(alpha_cavas)
    
    canvas = np.dstack((rgb_cavas, mask_cavas, overlay_cavas))

    return(canvas)
    
def generate_config_for_training(config_name, folder_name, hyperparameters, test_folder=None):

    config_path = os.path.join(ROOT_DIR, 'mmdetection', 'configs', 'romafo', config_name)
    
    with open(config_path, 'r') as file:
        config_content = file.readlines()
    
    data_root = os.path.join('..', 'data', 'synthetic_images', folder_name)
    work_dir = os.path.join('..', 'results', config_name.split('_')[0] + '_' + folder_name)
    
    myvar_optim_wd = hyperparameters['myvar_optim_wd']
    max_epochs = hyperparameters['max_epochs']
    stag_epochs = hyperparameters['stag_epochs']
    interval = hyperparameters['interval']
    batch_size = hyperparameters['batch_size']
    beginn_epochs_cosin_lr = hyperparameters['beginn_epochs_cosin_lr']
    end_iters_linear_lr = hyperparameters['end_iters_linear_lr']
    
    for i, line in enumerate(config_content):
        if 'DATA_ROOT' in line:
            config_content[i] = f"DATA_ROOT = '{data_root}{os.sep}'\n"
        elif 'work_dir' in line:
            config_content[i] = f"work_dir = '{work_dir}'\n"
        elif 'TEST_FOLDER' in line:
            if test_folder is not None:
                config_content[i] = f"TEST_FOLDER = '{test_folder}'\n"
        
        # here for writing hyperparameters
        elif 'MYVAR_OPTIM_WD' in line:
            config_content[i] = f"MYVAR_OPTIM_WD = {myvar_optim_wd}\n"
        elif 'MAX_EPOCHS' in line:
            config_content[i] = f"MAX_EPOCHS = {max_epochs}\n"
        elif 'STAG_EPOCHS' in line:
            config_content[i] = f"STAG_EPOCHS = {stag_epochs}\n"
        elif 'INTERVAL' in line:
            config_content[i] = f"INTERVAL = {interval}\n"
        elif 'BATCH_SIZE' in line:
            config_content[i] = f"BATCH_SIZE = {batch_size}\n"
        elif 'BEGINN_EPOCHS_COSIN_LR' in line:
            config_content[i] = f"BEGINN_EPOCHS_COSIN_LR = {beginn_epochs_cosin_lr}\n"
        elif 'END_ITERS_LINEAR_LR' in line:
            config_content[i] = f"END_ITERS_LINEAR_LR = {end_iters_linear_lr}\n"
        
        # only write first 15 line, here will be improved
        if i >= 20:
            break  
      
    with open(config_path, 'w') as file:
        file.writelines(config_content)
