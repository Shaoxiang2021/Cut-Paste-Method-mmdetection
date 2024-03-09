from cut import auto_cut_object
from processing import ImageGenerator
from config import *

if __name__ == '__main__':

    if cut_paste_mmdetection == 1:

        auto_cut_object(**config_cut_parameters)

    elif cut_paste_mmdetection == 2:

        imageGnerator = ImageGenerator(**config_paste_parameters)
        imageGnerator.image_generation()
