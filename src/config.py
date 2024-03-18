
############################### main parameter ###############################

# set the step 
# 1: cut step
# 2: only paste step
# 3: only train via mmdetection
# 4: only evaluation
# 5: train + evaluation
# 6: (demo) paste step + train + evaluation
cut_paste_mmdetection = 6

##################### 1. cut step - custom paramters #########################

# choose source objects
SOURCE = ['cylinder', 'plate', 'usb', 'fob', 'tempos']

# choose deep learning model to remove the background
# isnet-general-use: current best model
# u2net: general model
# u2netp: small fast model
SESSION = "isnet-general-use"

# --------------- no need to change here -------------------------------------

config_cut_parameters = {
    'source': SOURCE,
    'session': SESSION,
}

###############################################################################

##################### 2. past step - custom paramters #########################

# resolution for training
TRAIN_RESOLUTION = (720, 540)
# resolution of camera
CAMERA_RESOLUTION = (4032, 3024)

# set folder name 
# !!! suggestion! format for name: (number of class) _ (first letter of the class) _ (number of the images) _ (version) _ (info) !!!
FOLDER_NAME = "5_cpuft_1000_5"

# how many images should be generated (int)
NUM_IMAGES = 1000


# ---- if you want to change classes or parameters of the data augmentation ----

config_paste_parameters = {

    # general parameters

    # training classes (dict)
    'obj_dic': {
        'cylinder': 1,
        'plate': 2,
        'usb': 3,
        'fob': 4,
        'tempos': 5,
        },

    # how many images should be generated (int)
    'num_images': NUM_IMAGES,

    # format for name: number of class _ first letter of the class _ number of the images _ version _ info
    'folder_name': FOLDER_NAME, 

    # image size for training data (int)
    'size_x': TRAIN_RESOLUTION[0],
    'size_y': TRAIN_RESOLUTION[1],

    # how many objects should be in one single image (int)
    # min_obj < n < max_obj
    'min_obj': 4,
    'max_obj': 12,

    # set seed for torch random (int)
    'seed': 1,

    # how many times will the sw try to add object in every single image (int)
    'max_iter': 5,

    # probability for the overlapping (float) (0.0 - 1.0)
    # 0: there is no overlapping
    # 1: there is no limit for adding object in one single image
    'overlay_factor': 0.05,

    # value for generation_strategie (str)
    # NORMAL: generate same class, but different objects in single image
    # MIX: generate different classes and different objects in single image
    'generation_strategy': 'MIX',

    # value for annotation_strategie (str)
    # NORMAL: generate coco-labels, overlapped object with overlapped mask
    # SPECIAL: generate labels, overlapped object but with full mask
    'annotation_strategy': 'NORMAL',

    # generate mask for training data and validation data (bool)
    'generate_mask': False,

    # parameters for data augmentation

    'aug_params': {

        # roi is sometimes a little bit small, can be added more px for width and height (int)
        'correct_factor': 5,

        # value for scale_strategie (str)
        # NORMAL: resize object according to camera size and every objects in one single image with same scale
        # MIX: resize object according to camera size and every objects in one single image with different scale
        # above can be applyed for cuting images with different object size
        # ORIGINAL_MIX: every objects in one single image with different scale, but without resize object first
        'scale_strategy': 'MIX',

        # for resize step in the MIX and NORMAL scale_strategie
        'resize_in_procent': False,

        # --- using absolute px of width in camera resolution
        # object size in camera image (dict)
        # how big are the objects in camera resolution ? in px
        'obj_dic': {
        'cylinder': (1329, 544),
        'plate':    (747, 764),
        'usb':      (572, 164),
        'fob':      (571, 203),
        'tempos':   (1062, 421),
        },
        
        # --- using procent factor of width in camera resolution
        # object size in camera image (float)
        # !!! keep_ratio must be True !!!
        # how big are the objects in camera resolution ? estimate the size of the target in procent # 0.21 0.1 0.075 0.09 0.22

        #'obj_dic': {
        #'cylinder': 0.26,
        #'plate':    0.13,
        #'usb':      0.13,
        #'fob':      0.12,
        #'tempos':   0.265,
        #},
        
        # resolution for camera and training data
        'camera_resolution': CAMERA_RESOLUTION,
        'train_resolution': TRAIN_RESOLUTION,
        
        # keep ratio of height and width
        'keep_ratio': True,

        # augmentation parameters for torch 
        'shadow_strength': 0.25,
        'min_scale_factor': 0.8, 
        'max_scale_factor': 1.2,
        'max_degrees': 180, 
        'distortion_scale': 0.2,
        
        'color_jitter': {
            'brightness': (1.0, 1.4), 
            'saturation': 0.15, 
            'hue': 0.15,
            },

        'gaussian_blur':{
            'kernel_size': (7, 7), 
            'sigma': (0.1, 1.0),
        },

        'general_probability': 0.5,
        }
}

###############################################################################

################# 3. training step - custom paramters #########################

# config file name in mmetection/configs/romafo/
# crcnn: cascade-mask-rcnn_r50_fpn_1x_{num_images}.py
# solov2: solov2_r50_fpn_ms-3x_{num_images}.py
# rtmdet: rtmdet-ins_m_8xb32-300e_coco_{num_images}.py
# yolact: yolact_r50_1x8_coco_{num_images}.py
# spareinst: sparseinst_r50_iam_8xb8_ms-270k_coco_{num_images}.py (comming soon)

CONFIG_NAMES = ['rtmdet-ins_m_8xb32-300e_coco_1000.py']

###############################################################################

############# 4. evaluation step - custom parameters ##########################

# folder path for inferece 
INFERENCE_FOLDER_NAME = 'demo_1'

# folder path for testing
TEST_FOLDER_NAMES = ['test_1', 'test_cylinder', 'test_plate', 'test_usb']
