
# load custom files
# custom_imports = dict(imports=["custom.augmentation"], allow_failed_imports=False)

# env parameters
MYVAR_OPTIM_LR = 1e-05
MYVAR_OPTIM_WD = 0.0001
MAX_EPOCHS = 10
BATCH_SIZE = 8
DATA_ROOT = '../data/synthetic_images/5_cpuft_1000_1/'

load_from = "../mmdetection/checkpoints/sparseinst_r50_iam_8xb8-ms-270k_coco_20221111_181051-72c711cd.pth"
work_dir = '../results/sparseinst_5_cpuft_1000_1'

_base_ = '../../projects/SparseInst/configs/sparseinst_r50_iam_8xb8-ms-270k_coco_1000.py'