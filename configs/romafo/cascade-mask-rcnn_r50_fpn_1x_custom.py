# env parameters
MYVAR_OPTIM_LR = 1e-05
MYVAR_OPTIM_WD = 0.0001
MAX_EPOCHS = 10
BATCH_SIZE = 4
DATA_ROOT = '../data/synthetic_images/5_cpuft_1000_1/'

TEST_FOLDER = 'test_1'
TEST_ROOT = '../data/source_images/05_test/test/'
load_from = "../mmdetection/checkpoints/cascade_mask_rcnn_r50_caffe_fpn_1x_coco_bbox_mAP-0.412__segm_mAP-0.36_20200504_174659-5004b251.pth"
work_dir = '../results/cascade-mask-rcnn_5_cpuft_1000_1'



_base_ = '../cascade_rcnn/cascade-mask-rcnn_r50-caffe_fpn_1x_coco.py'

# model parameters
model = dict(roi_head=dict(bbox_head=[dict(num_classes=5, type='Shared2FCBBoxHead'), dict(num_classes=5, type='Shared2FCBBoxHead'), dict(num_classes=5, type='Shared2FCBBoxHead')], mask_head=dict(num_classes=5, type='FCNMaskHead')))

# set classes
metainfo = dict(classes=('cylinder', 'plate', 'usb', 'fob', 'tempos'))

# optimizer
# optim_wrapper = dict(
#    clip_grad=dict(max_norm=35, norm_type=2),
#    optimizer=dict(lr=MYVAR_OPTIM_LR, momentum=0.9, type='SGD', weight_decay=0.0001),
#    type='OptimWrapper')

optim_wrapper = dict(type='OptimWrapper', optimizer = dict(_delete_=True, type='Adam', lr=MYVAR_OPTIM_LR, weight_decay=MYVAR_OPTIM_WD), clip_grad=None)

# param_scheduler
# set the parameters need to see the training curve, LinearLR or MultiStepLR
# param_scheduler = dict(_delete_=True, type='MultiStepLR', by_epoch=True, begin=0, end=10, milestones=[5, 10, 20],  gamma=0.1)
# param_scheduler = [dict(type='LinearLR', start_factor=1e-07, by_epoch=False, begin=0, end=200), dict(type='CosineAnnealingLR', T_max=6050, by_epoch=False, begin=200, end=6250)]
param_scheduler = [dict(type='LinearLR', start_factor=1e-07, by_epoch=False, begin=0, end=100), dict(type='CosineAnnealingLR', T_max=2400, by_epoch=False, begin=100, end=2500)]

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
