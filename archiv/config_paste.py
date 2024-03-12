# resolution for training
TRAIN_RESOLUTION = (720, 540)
# resolution of camera
CAMERA_RESOLUTION = (1280, 960)

config_parameters = {

    # general parameters

    # training classes (dict)
    'obj_dic': {
        'cylinder': 1,
        'plate': 2,
        'usb': 3,
        'fob': 4,
        'tempos': 5,
        },

    # root path for the source images and output path (str)
    'source_root': "/home/tan/MA_Shaoxiang/data/synthetic_images", ##### test data augmentation in Trainig ##### color jitter and gaussian blur
    'output_root': "/home/tan/MA_Shaoxiang/data/synthetic_images/5_cpuft_5000_ca_2", # number of class _ first letter of the class _ number of the images _ version

    # how many images should be generated (int)
    'num_images': 5000,

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
    'overlay_factor': 0.1,

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
        'resize_in_procent': True,

        # --- using absolute px of width in camera resolution
        # object size in camera image (dict)
        # how big are the objects in camera resolution ? in px
        #'obj_dic': {
        #'cylinder': (268, 113),
        #'plate':    (120, 127),
        #'usb':      (94, 29),
        #'fob':      (94, 29),
        #'tempos':   (268, 113),
        #},
        
        # --- using procent factor of width in camera resolution
        # object size in camera image (float)
        # !!! keep_ratio must be True !!!
        # how big are the objects in camera resolution ? estimate the size of the target in procent

        'obj_dic': {
        'cylinder': 0.21,
        'plate':    0.1,
        'usb':      0.075,
        'fob':      0.09,
        'tempos':   0.21,
        },
        
        # resolution for camera and training data
        'camera_resolution': CAMERA_RESOLUTION,
        'train_resolution': TRAIN_RESOLUTION,
        
        # keep ratio of height and width
        'keep_ratio': True,

        # augmentation parameters for torch 
        'shadow_strength': 0.25,
        'min_scale_factor': 0.9, 
        'max_scale_factor': 1.3,
        'max_degrees': 180, 
        'distortion_scale': 0.2,

        'general_probability': 0.5,
        }
}

