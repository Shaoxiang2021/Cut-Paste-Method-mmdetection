
from mmdet.apis import DetInferencer
from path import ROOT_DIR
import os

def demo_prediction(inference_folder, folder_name, config_name):

    folder_path = os.path.join(ROOT_DIR, 'data', 'source_images', '05_test', 'test', inference_folder) + os.sep
    model_output_path = os.path.join(ROOT_DIR, 'results', config_name.split('_')[0] + '_' + folder_name)
    model_path = os.path.join(ROOT_DIR, 'results', config_name.split('_')[0] + '_' + folder_name, 'epoch_10.pth')
    
    # using copied training config
    config_path = os.path.join(ROOT_DIR, 'mmdetection', 'configs', 'romafo', config_name)
    # config_path = os.path.join(ROOT_DIR, 'results', config_name.split('_')[0] + '_' + folder_name, config_name)
    
    inferencer = DetInferencer(model=config_path, weights=model_path, show_progress=True, palette='none')
    inferencer(inputs=folder_path, out_dir=model_output_path, no_save_pred=False, pred_score_thr=0.3)
    
def evaluate_via_test_data(folder_name, config_name, epoch, save_pred=False):

    tool_path = os.path.join(ROOT_DIR, 'mmdetection', 'tools', 'test.py')
    
    # using copied training config
    # config_path = os.path.join(ROOT_DIR, 'mmdetection', 'configs', 'romafo', config_name)
    config_path = os.path.join(ROOT_DIR, 'results', config_name.split('_')[0] + '_' + folder_name, config_name)
    
    # epoch is from sw config not from training config
    model_path = os.path.join(ROOT_DIR, 'results', config_name.split('_')[0] + '_' + folder_name, 'epoch_{}.pth'.format(epoch))
    
    output_path = os.path.join(ROOT_DIR, 'results', config_name.split('_')[0] + '_' + folder_name, 'test.pkl')
    
    if save_pred is not True:
        command_test = 'python' + ' ' + tool_path + ' ' + config_path + ' ' + model_path + ' ' + '--out' + ' ' + output_path
        os.system(command_test)
    else:
        command_test = 'python' + ' ' + tool_path + ' ' + config_path + ' ' + model_path + ' ' + '--out' + ' ' + output_path + ' ' + '--show-dir' + ' ' + 'vis_results' 
        os.system(command_test)
