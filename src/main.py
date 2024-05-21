
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
        self.test_folders = TEST_FOLDER_NAMES
        self.inference_folder = INFERENCE_FOLDER_NAME
        
        self.config_cut_parameters = config_cut_parameters
        self.config_paste_parameters = config_paste_parameters
        self.step = cut_paste_mmdetection
        self.save_pred = save_pred
        
        self.hyperparameters = dict(
                                    myvar_optim_wd = MYVAR_OPTIM_WD,
                                    max_epochs = MAX_EPOCHS, 
                                    stag_epochs = STAG_EPOCHS, 
                                    interval = INTERVAL, 
                                    batch_size = BATCH_SIZE, 
                                    beginn_epochs_cosin_lr = BEGINN_EPOCHS_COSIN_LR,
                                    end_iters_linear_lr = END_ITERS_LINEAR_LR,
                                    )
        
    def cut(self):
    
        auto_cut_object(**self.config_cut_parameters)
    
    def paste(self):
    
        imageGnerator = ImageGenerator(**self.config_paste_parameters)
        imageGnerator.image_generation()
        
    def train(self):
        
        for config_name in self.config_names:
        
            print("generate config file for training")
            generate_config_for_training(config_name, self.folder_name, self.hyperparameters, None)
            
            print("start to train ...")
            command_train = "python" + " " + os.path.join(ROOT_DIR, 'mmdetection', 'tools', 'train.py') + " " + os.path.join(ROOT_DIR, 'mmdetection', 'configs', 'romafo', config_name) 
            os.system(command_train)
    
    def inference(self):
        
        for config_name in self.config_names:
        
            print("generate config file for training")
            generate_config_for_training(config_name, self.folder_name, self.hyperparameters, self.inference_folder)
        
            print("model inferencing ...")
            demo_prediction(self.inference_folder, self.folder_name, config_name)
            
    def evaluate(self):
    
        for config_name in self.config_names:
        
            for test_folder in self.test_folders:
        
                print("generate config file for training")
                generate_config_for_training(config_name, self.folder_name, self.hyperparameters, test_folder)
                
                print("model testing ...")
                evaluate_via_test_data(self.folder_name, config_name, self.hyperparameters['max_epochs'], save_pred=self.save_pred)
          
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
            
            self.inference()
            
        elif self.step == 6:
            
            #try:
                #self.train()
                #self.evaluate()
            
            #except Exception as e:
                #print("An error occurred:", e)
            
            self.train()
            self.evaluate()
        
        elif self.step == 7:
            
            #try:
                #self.paste()
                #self.train()
                #self.evaluate()
                
            #except Exception as e:
                #print("An error occurred:", e)
                
            self.paste()
            self.train()
            self.evaluate()
                
if __name__ == '__main__':

    processor = Processor()
    
    processor()
            
