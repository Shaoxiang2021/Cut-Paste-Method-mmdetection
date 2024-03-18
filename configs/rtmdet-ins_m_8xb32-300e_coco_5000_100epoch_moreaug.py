# env parameters
MYVAR_OPTIM_LR = 1e-03
MYVAR_OPTIM_WD = 0.0001
MAX_EPOCHS = 100
STAG_EPOCHS = 20
INTERVAL = 10
BATCH_SIZE = 8
DATA_ROOT = '../data/synthetic_images/5_cpuft_5000_1/'

TEST_FOLDER = 'test_usb'
TEST_ROOT = '../data/source_images/05_test/test/'
load_from = "../mmdetection/checkpoints/rtmdet-ins_m_8xb32-300e_coco_20221123_001039-6eba602e.pth"
work_dir = '../results/rtmdet-ins_5_cpuft_5000_1'

_base_ = '../rtmdet/rtmdet-ins_m_8xb32-300e_coco.py'

default_hooks = dict(
    checkpoint=dict(
        interval=INTERVAL,
        max_keep_ckpts=3  # only keep latest 3 checkpoints
    ))

# model parameters
model = dict(backbone=dict(norm_cfg=dict(type='BN')), bbox_head=dict(num_classes=5, norm_cfg=dict(requires_grad=True, type='BN')), neck=dict(norm_cfg=dict(type='BN')))
# model = dict(bbox_head=dict(num_classes=5))

# set classes
metainfo = dict(classes=('cylinder', 'plate', 'usb', 'fob', 'tempos'), palatte=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)])

# fp16 = dict(loss_scale=512.) 
# optimizer
# optim_wrapper = dict(
#    clip_grad=dict(max_norm=35, norm_type=2),
#    optimizer=dict(lr=MYVAR_OPTIM_LR, momentum=0.9, type='SGD', weight_decay=0.0001),
#    type='OptimWrapper')

optim_wrapper = dict(_delete_=True, type='OptimWrapper', optimizer=dict(type='AdamW', lr=MYVAR_OPTIM_LR, weight_decay=MYVAR_OPTIM_WD), clip_grad=None)

# param_scheduler
# set the parameters need to see the training curve, LinearLR or MultiStepLR
# param_scheduler = dict(_delete_=True, type='MultiStepLR', by_epoch=True, begin=0, end=10, milestones=[5, 10, 20],  gamma=0.1)
# param_scheduler = [dict(type='LinearLR', start_factor=1e-07, by_epoch=False, begin=0, end=200), dict(type='CosineAnnealingLR', T_max=6050, by_epoch=False, begin=200, end=6250)]
param_scheduler = [dict(type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0, end=1000), dict(type='CosineAnnealingLR', T_max=MAX_EPOCHS, by_epoch=True, convert_to_iter_based=True)]

# pipline for training, validation and test
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(type='RandomResize', scale=(1280, 1280), ratio_range=(0.1, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    #dict(type='PhotoMetricDistortion', contrast_range=(0.75, 1.25), saturation_range=(0.75, 1.25)),
    #dict(type='Sharpness', min_mag=0.5, max_mag=1.5),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs'),
]

train_pipeline_stage_II = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=(640, 640),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
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

# train dataloader parameters
train_cfg = dict(_delete_=True, max_epochs=MAX_EPOCHS, val_interval=INTERVAL, type='EpochBasedTrainLoop', dynamic_intervals=[(MAX_EPOCHS - STAG_EPOCHS, 1)])

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    pin_memory=True,
    dataset=dict(
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/imgs/'),
        data_root=DATA_ROOT,
        pipeline=train_pipeline,
        metainfo=metainfo)
        )

# test dataloader parameters
test_dataloader = dict(
    batch_size=1,
    dataset = dict(
        ann_file=TEST_FOLDER+'/annotations.json',
        data_prefix=dict(img=TEST_FOLDER+'/'),
        data_root=TEST_ROOT,
        pipeline=test_pipeline,
        metainfo=metainfo)
        )

test_evaluator = dict(
    ann_file=TEST_ROOT+TEST_FOLDER+'/annotations.json',
    metric='segm',
    type='CocoMetric',
    outfile_prefix=work_dir)

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

custom_hooks = [
    dict(
        type='PipelineSwitchHook',
        switch_epoch=MAX_EPOCHS - STAG_EPOCHS,
        switch_pipeline=train_pipeline_stage_II)
]
