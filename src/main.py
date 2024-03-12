
from config import *
from cut import auto_cut_object
from processing import ImageGenerator
from handling import generate_config_for_training
from path import ROOT_DIR
from evaluation import *
import os
import shutil

class Processor(object):

    def __init__(self):
        
        self.folder_name = FOLDER_NAME
        self.config_names = CONFIG_NAMES
        self.test_folder = TEST_FOLDER_NAME
        
        self.config_cut_parameters = config_cut_parameters
        self.config_paste_parameters = config_paste_parameters
        
        self.step = cut_paste_mmdetection
        
    def cut(self):
    
        auto_cut_object(**self.config_cut_parameters)
    
    def paste(self):
    
        imageGnerator = ImageGenerator(**self.config_paste_parameters)
        imageGnerator.image_generation()
        
    def train(self):
        
        for config_name in self.config_names:
        
            print("generate config file for training")
            generate_config_for_training(config_name, self.folder_name)
            
            print("start to train ...")
            command_train = "python" + " " + os.path.join(ROOT_DIR, 'mmdetection', 'tools', 'train.py') + " " + os.path.join(ROOT_DIR, 'mmdetection', 'configs', 'romafo', config_name) 
            os.system(command_train)
    
    def evaluate(self):
        
        for config_name in self.config_names:
        
            print("model inferencing ...")
            demo_prediction(self.test_folder, self.folder_name, config_name)
          
    def __call__(self):
    

        if self.step == 1:
    
            self.cut()
    
        elif self.step == 2:
    
            self.paste()
        
        elif self.step == 3:
            
            self.train()
            
        elif self.step == 4:
            
            self.evaluate()
        
        elif self.step == 5:
            
            """
            try:
                self.paste()
                self.train()
                self.evaluate()
                
            except Exception as e:
                print("An error occurred:", e)
            """
            
            self.paste()
            self.train()
            self.evaluate()
                
if __name__ == '__main__':

    processor = Processor()
    
    processor()
            
