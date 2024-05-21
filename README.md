# CP-mmdetection Demo

This demo showcases how to automatically generate training data through the cut-paste method, as well as train and evaluate models for industrial applications. You can only take few photos for the target object and then run the software. After two hours you can get a excellent cnn model for the instance segmentation use case. This Software is currently only used for solid object, for not solid object is the performance not so well. 

Following image show the example results of our research:

 -- SoloV2
![industry object result 1](readme/test_40.png)
 -- rtmdet 
![industry object result 1](readme/test_60.png)

## Table of contents   
- [(1) Environment](#(1)-Environment) 
- [(2) Software Structure](#(2)-Software-Structure) 
- [(3) Run The Software](#(3)-Run-The-Software) 

## (1) Environment

1. First, you need to download and install Conda. Both Miniconda and Anaconda are suitable options for this. More you can see: https://www.anaconda.com/. For example in Linux: 
    ```
    bash Miniconda3-latest-Linux-x86_64.sh
    ```

2. The training section of this demo is based on MMDetection, so it's crucial to correctly install the MMDetection framework. You can follow the steps outlined in the documentation https://mmdetection.readthedocs.io/en/latest/get_started.html for installation, create conda environment and use the provided test code to verify if the configuration was successful. Here, it's important to match the GPU driver with the CUDA version, more you can see: https://pytorch.org/get-started/locally/ please choose conda install. 

3. Other required packages you can simply install using a requirements.txt file.
    ```
    pip install -r requirements.txt
    ```

## (2) Software Structure

```
project
│   README.md
│   requirements.txt
│
└───data
│   │    
│   └───source_images
│   │   - 01_canvas
│   │   - 02_raw
│   │   - 03_cut
│   │   - 04_crop
│   │   - 05_test
│   └───synthetic_images
│       - subfolders
│       - ...
│   
└───mmdetection
└───results
└───src
│   │    
│   └───source_images
│   │   - __init__.py
│   │   - annotation.py
│   │   - augmentation.py
│   │   - config.py
│   │   - cut.py
│   │   - evaluation.py
│   │   - handing.py
│   │   - main.py
│   │   - path.py
│   │   - processing.py
│   
└─────
```

## (3) Running the Software

To run the software, you only need to configure the config file and then execute the main program. 

    cd {your project path}/src
    conda activate {your environment name}
    python main.py
    
before you run the main program, you need make sure required data also in the right path. 

### (3.1) Cut-Step

In this process, you need to copy your target image files into the "02_raw" directory, for example, from a USB stick. Create a new folder named "usb" within "02_raw" and paste all the source images into it. 

During the program's execution, it will save the cropped images along with their corresponding masks into the "03_cut/{target}" directory. 

### (3.2) Paste-Step

For the paste step, you need to select images without error masks and copy them into the corresponding folders within the "04_crop/{target}" directory.

The generated data and COCO labels will be stored in the "synthetic_images/{folder name}" folder.

### (3.3) Training-Step

If you use pre-trained model, you need first to download model from https://mmdetection.readthedocs.io/en/latest/model_zoo.html, and copy it to "mmdetection/checkpoints/". Please select in config.py the right config files for the training. 

The results will be automatically stored in the "results/{folder name}" folder.

### (3.4) Evaluation-Step

Copy your test images in "05_test/demo/{folder name}" folder. 

[(back to top)](#table-of-contents)






## Authors
Shaoxiang Tan (ge28kiw@tum.de)
