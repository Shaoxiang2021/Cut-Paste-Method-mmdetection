
# load custom files
# custom_imports = dict(imports=["custom.augmentation"], allow_failed_imports=False)

# env parameters
MYVAR_OPTIM_LR = 1e-05
MYVAR_OPTIM_WD = 0.0001
MAX_EPOCHS = 10
BATCH_SIZE = 8
DATA_ROOT = '../data/synthetic_images/5_cpuft_5000_1/'

load_from = "../mmdetection/checkpoints/solov2_r50_fpn_3x_coco_20220512_125856-fed092d4.pth"
work_dir = '../results/solov2_5_cpuft_5000_1'

_base_ = '../solov2/solov2_r50_fpn_ms-3x_coco.py'

# model parameters
model = dict(mask_head=dict(num_classes=5))

# set classes
metainfo = dict(classes=('cylinder', 'plate', 'usb', 'fob', 'tempos'), palatte=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)])

# optimizer

# optim_wrapper = dict(
#    clip_grad=dict(max_norm=35, norm_type=2),
#    optimizer=dict(lr=MYVAR_OPTIM_LR, momentum=0.9, type='SGD', weight_decay=0.0001),
#    type='OptimWrapper')

optim_wrapper = dict(_delete_=True, type='OptimWrapper', optimizer = dict(type='Adam', lr=MYVAR_OPTIM_LR, weight_decay=MYVAR_OPTIM_WD), clip_grad=None)

# param_scheduler
# set the parameters need to see the training curve, LinearLR or MultiStepLR
# param_scheduler = dict(_delete_=True, type='MultiStepLR', by_epoch=True, begin=0, end=10, milestones=[5, 10, 20],  gamma=0.1)
param_scheduler = [dict(type='LinearLR', start_factor=1e-07, by_epoch=False, begin=0, end=200), dict(type='CosineAnnealingLR', T_max=6050, by_epoch=False, begin=200, end=6250)]
# param_scheduler = [dict(type='LinearLR', start_factor=1e-07, by_epoch=False, begin=0, end=50), dict(type='CosineAnnealingLR', T_max=1200, by_epoch=False, begin=50, end=1250)]

# pipline for training, validation and test
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(keep_ratio=True, scale=(1333,800), type='Resize'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion', contrast_range=(0.75, 1.25), saturation_range=(0.75, 1.25)),
    dict(type='Sharpness', min_mag=0.5, max_mag=1.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(1333,800), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

val_pipeline = test_pipeline

# test dataloader parameters
test_dataloader = dict(
    batch_size=1,
    dataset = dict(
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/imgs/'),
        data_root=DATA_ROOT,
        pipeline=test_pipeline,
        metainfo=metainfo)
        )

test_evaluator = dict(
    ann_file=DATA_ROOT+'test/annotations.json',
    metric='segm',
    type='CocoMetric',
    outfile_prefix='results/romafo/solov2_r50_cylinder_overlap/test')

# train dataloader parameters
train_cfg = dict(max_epochs=MAX_EPOCHS, type='EpochBasedTrainLoop', val_interval=1)

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    dataset=dict(
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/imgs/'),
        data_root=DATA_ROOT,
        pipeline=train_pipeline,
        metainfo=metainfo)
        )

# validation dataloader parameters
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/imgs/'),
        data_root=DATA_ROOT,
        pipeline=val_pipeline,
        metainfo=metainfo)
        )

val_evaluator = dict(
    ann_file=DATA_ROOT+'val/annotations.json',
    metric='segm',
    type='CocoMetric')
