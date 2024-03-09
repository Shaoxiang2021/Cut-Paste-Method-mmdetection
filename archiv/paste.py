from processing import ImageGenerator
from config import config_paste_parameters

if __name__ == '__main__':

    imageGnerator = ImageGenerator(**config_paste_parameters)
    imageGnerator.image_generation()
    