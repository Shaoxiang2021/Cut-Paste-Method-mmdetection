from cut import auto_cut_object
from processing import ImageGenerator
from config import *
from handling import generate_config_for_training
from path import ROOT_DIR
from evaluation import *
import os
import shutil

def cut():
    
    auto_cut_object(**config_cut_parameters)
    
def paste():

    imageGnerator = ImageGenerator(**config_paste_parameters)
    imageGnerator.image_generation()
    
def train():
    
    for CONFIG_NAME in CONFIG_NAMES:
    
      print("generate config file for training")
      generate_config_for_training(CONFIG_NAME, FOLDER_NAME)
      
      print("start to train ...")
      command_train = "python" + " " + os.path.join(ROOT_DIR, 'mmdetection', 'tools', 'train.py') + " " + os.path.join(ROOT_DIR, 'mmdetection', 'configs', 'romafo', CONFIG_NAME) 
      os.system(command_train)
    
    #shutil.copy(os.path.join(ROOT_DIR, 'mmdetection', 'configs', 'romafo', CONFIG_NAME), os.path.join('..', 'results', CONFIG_NAME.split('_')[0] + '_' + FOLDER_NAME))
    
def evaluate():
    
    print("model inferencing ...")
    demo_prediction()

if __name__ == '__main__':

    if cut_paste_mmdetection == 1:

        cut()

    elif cut_paste_mmdetection == 2:

        paste()
    
    elif cut_paste_mmdetection == 3:
        
        train()
        
    elif cut_paste_mmdetection == 4:
        
        evaluate()
    
    elif cut_paste_mmdetection == 5:
        
        try:
            paste()
            train()
            evaluate()
            
        except Exception as e:
            print("An error occurred:", e)
        