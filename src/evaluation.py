
from mmdet.apis import DetInferencer
from path import ROOT_DIR
import os

def demo_prediction(test_folder_name, folder_name, config_name):

    folder_path = os.path.join(ROOT_DIR, 'data', 'source_images', '05_test', 'demo', test_folder_name) + os.sep
    model_output_path = os.path.join('..', 'results', config_name.split('_')[0] + '_' + folder_name)
    model_path = os.path.join(ROOT_DIR, 'results', config_name.split('_')[0] + '_' + folder_name, 'epoch_10.pth')
    config_path = os.path.join(ROOT_DIR, 'mmdetection', 'configs', 'romafo', config_name)
    
    inferencer = DetInferencer(model=config_path, weights=model_path, scope='mmdet', show_progress=True)
    
    inferencer(inputs=folder_path, out_dir=model_output_path, draw_pred=True, no_save_pred=False)
