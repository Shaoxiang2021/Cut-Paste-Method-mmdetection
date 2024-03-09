import os
from PIL import Image
from pathlib import Path
from rembg import remove
from rembg.session_factory import new_session
from tqdm import tqdm

from config import config_cut_parameters
from path import ROOT_DIR

def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    image_names = []
    file_extensions = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_names.append(file)
                file_extensions.append(Path(file).suffix)

    return image_names, file_extensions

def auto_cut_object(source, session):

    for cls in source:

        folder_path = os.path.join(ROOT_DIR, 'data', 'source_images', '02_raw', cls)
        image_names, file_extensions = get_image_paths(folder_path)

        cut_path = os.path.join(ROOT_DIR, 'data', 'source_images', '03_cut', cls)

        if not os.path.exists(cut_path):
            os.makedirs(cut_path)

        print('processing for target {} ...'.format(cls))
        
        for index, image_name in tqdm(enumerate(image_names), total=len(image_names)):

            original_image = Image.open(os.path.join(folder_path, image_name))

            image_removebg = remove(original_image, session=new_session(session), bgcolor=None, post_process_mask=True)
            mask = remove(original_image, session=new_session(session), only_mask=True, post_process_mask=True)
            
            output_image_path = os.path.join(cut_path, image_name.rstrip(file_extensions[index])+".png")
            output_mask_path = os.path.join(cut_path, image_name.rstrip(file_extensions[index])+"_mask.png")

            image_removebg.save(output_image_path)
            mask.save(output_mask_path)
