
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
SOURCE = ['fob', 'tempos']

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
CAMERA_RESOLUTION = (1280, 960)

# class for the generation
OBJ_LIST = ['tempos', 'fob', 'mars', 'envelope']

# for resize step in the MIX and NORMAL scale_strategie
RESIZE_IN_PROCENT = False

# if you set resize in procent in true
OBJ_DIC = {
          'tempos':    (210, 101),
          'fob':       (115, 40),
          'mars':      (220, 61),
          'envelope':  (420, 287),
}

# if add distractors is needed
ADD_DISTRACTORS = False

DIST_LIST = ['disk', 'metal', 'mount', 'plastic', 'ring', 'seal']
DIST_DIC = {
            'disk':    (199, 192),
            'metal':   (255, 181),
            'mount':   (487, 333),
            'plastic': (537, 542),
            'ring':    (313, 305),
            'seal':    (266, 263),
}

# set folder name 
# !!! suggestion! format for name: (number of class) _ (first letter of the class) _ (number of the images) _ (version) _ (info) !!!
FOLDER_NAME = "4_tfme_1000_1_0"

# how many images should be generated (int)
NUM_IMAGES = 1000

# hook for generation strategy
HOOK = {
200: {'min_obj': 2, 'max_obj': 8, 'overlay_factor': 0.5, 'use_dist': False, 'generation_strategy': 'MIX', 'scale_strategy': 'MIX'},
400: {'min_obj': 12, 'max_obj': 20, 'overlay_factor': 1, 'use_dist': False, 'generation_strategy': 'MIX', 'scale_strategy': 'MIX'},
800: {'min_obj': 2, 'max_obj': 8, 'overlay_factor': 0, 'use_dist': False, 'generation_strategy': 'MIX', 'scale_strategy': 'MIX'},
}

# ---- if you want to change classes or parameters of the data augmentation ----

config_paste_parameters = {

    # general parameters

    # training classes (dict)
    'obj_list': OBJ_LIST,

    'use_dist': ADD_DISTRACTORS,
    'dist_list': DIST_LIST,
    
    'min_dist': 6, 
    'max_dist': 12,

    # how many images should be generated (int)
    'num_images': NUM_IMAGES,

    # format for name: number of class _ first letter of the class _ number of the images _ version _ info
    'folder_name': FOLDER_NAME, 

    # image size for training data (int)
    'size_x': TRAIN_RESOLUTION[0],
    'size_y': TRAIN_RESOLUTION[1],

    # how many objects should be in one single image (int)
    # min_obj < n < max_obj
    'min_obj': 2,
    'max_obj': 8,

    # set seed for torch random (int)
    'seed': 1,

    # how many times will the sw try to add object in every single image (int)
    'max_iter': 5,

    # probability for the overlapping (float) (0.0 - 1.0)
    # 0: there is no overlapping
    # 1: there is no limit for adding object in one single image
    'overlay_factor': 0,

    # value for generation_strategie (str)
    # NORMAL: generate same class, but different objects in single image
    # MIX: generate different classes and different objects in single image
    'generation_strategy': 'MIX',

    # value for annotation_strategie (str)
    # NORMAL: generate coco-labels, overlapped object with overlapped mask
    # SPECIAL: generate labels, overlapped object but with full mask
    'annotation_strategy': 'NORMAL',
    
    # value for scale_strategie (str)
    # NORMAL: resize object according to camera size and every objects in one single image with same scale
    # MIX: resize object according to camera size and every objects in one single image with different scale
    # above can be applyed for cuting images with different object size
    # ORIGINAL_MIX: every objects in one single image with different scale, but without resize object first (not use anymore)
    'scale_strategy': 'MIX',

    # generate mask for training data and validation data (bool)
    'generate_mask': False,
    
    # hook
    'hook': HOOK,

    # parameters for data augmentation

    'aug_params': {

        # roi is sometimes a little bit small, can be added more px for width and height (int)
        'correct_factor': 5,

        # value for scale_strategie (str)
        # NORMAL: resize object according to camera size and every objects in one single image with same scale
        # MIX: resize object according to camera size and every objects in one single image with different scale
        # above can be applyed for cuting images with different object size
        # ORIGINAL_MIX: every objects in one single image with different scale, but without resize object first
        #'scale_strategy': 'MIX',

        # for resize step in the MIX and NORMAL scale_strategie
        'resize_in_procent': RESIZE_IN_PROCENT,

        'obj_list': OBJ_LIST,

        'load_dist': False,
        'dist_list': DIST_LIST,

        # --- using absolute px of width in camera resolution
        # object size in camera image (dict)
        # how big are the objects in camera resolution ? in px
        'obj_dic': OBJ_DIC,
        'dist_dic': DIST_DIC,
        
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
            'brightness': (0.5, 1.5), 
            'saturation': 0.25, 
            'hue': 0.25,
            },

        'gaussian_blur':{
            'kernel_size': (7, 7), 
            'sigma': (0.1, 0.5),
        },

        'general_probability': 0.5,
        }
}

###############################################################################

################# 3. training step - custom paramters #########################

# config file name in mmetection/configs/romafo/
# (2018) cmrcnn: cascade-mask-rcnn_r50_fpn_1x_custom.py
# (2019) yolact: yolact_r50_1x8_coco_custom.py
# (2020) solov2: solov2_r50_fpn_ms-3x_custum.py
# (2022) spareinst: sparseinst_r50_iam_8xb8_ms-270k_coco_custom.py
# (2023) rtmdet: rtmdet-ins_m_8xb32-300e_coco_custom.py

# training hyperparameters
MYVAR_OPTIM_WD = 0.001
MAX_EPOCHS = 10
STAG_EPOCHS = 2
INTERVAL = 5
BATCH_SIZE = 4
BEGINN_EPOCHS_COSIN_LR = 5
END_ITERS_LINEAR_LR = 200

# config files put in the list
CONFIG_NAMES = ['rtmdet-ins_m_8xb32-300e_coco_custom.py']

###############################################################################

############# 4. evaluation step - custom parameters ##########################

# folder path for inferece 
INFERENCE_FOLDER_NAME = 'demo_1'

# folder path for testing
# test_1: example for general test in karmera resolution
# test_cylinder, test_plate, test_usb: Test dataset from paper in handy resolution
TEST_FOLDER_NAMES = ['test_ws_industry_objects', 'test_cylinder', 'test_plate', 'test_usb']
