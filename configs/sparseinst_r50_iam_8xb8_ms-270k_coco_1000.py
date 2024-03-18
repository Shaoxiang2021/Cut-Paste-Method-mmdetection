# env parameters
MYVAR_OPTIM_LR = 1e-05
MYVAR_OPTIM_WD = 0.0001
MAX_EPOCHS = 10
BATCH_SIZE = 8
DATA_ROOT = '../data/synthetic_images/5_cpuft_1000_1/'
NUM_CLASS = 5

TEST_FOLDER = 'test_1'
TEST_ROOT = '../data/source_images/05_test/test/'
load_from = "../mmdetection/checkpoints/sparseinst_r50_iam_8xb8-ms-270k_coco_20221111_181051-72c711cd.pth"
work_dir = '../results/sparseinst_5_cpuft_1000_1'


_base_ = [
    'mmdet::_base_/datasets/coco_instance.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['projects.SparseInst.sparseinst'], allow_failed_imports=False)
    

metainfo = dict(classes=('cylinder', 'plate', 'usb', 'fob', 'tempos'), palatte=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)])

model = dict(
    type='SparseInst',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    encoder=dict(
        type='InstanceContextEncoder',
        in_channels=[512, 1024, 2048],
        out_channels=256),
    decoder=dict(
        type='BaseIAMDecoder',
        in_channels=256 + 2,
        num_classes=NUM_CLASS,
        ins_dim=256,
        ins_conv=4,
        mask_dim=256,
        mask_conv=4,
        kernel_dim=128,
        scale_factor=2.0,
        output_iam=False,
        num_masks=100),
    criterion=dict(
        type='SparseInstCriterion',
        num_classes=NUM_CLASS,
        assigner=dict(type='SparseInstMatcher', alpha=0.8, beta=0.2),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            alpha=0.25,
            gamma=2.0,
            reduction='sum',
            loss_weight=2.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            reduction='sum',
            eps=5e-5,
            loss_weight=2.0),
    ),
    test_cfg=dict(score_thr=0.005, mask_thr_binary=0.45))

backend = 'pillow'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args={{_base_.backend_args}},
        imdecode_backend=backend),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(
        type='RandomChoiceResize',
        scales=[(416, 853), (448, 853), (480, 853), (512, 853), (544, 853),
                (576, 853), (608, 853), (640, 853)],
        keep_ratio=True,
        backend=backend),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    #dict(type='Sharpness', min_mag=0.5, max_mag=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args={{_base_.backend_args}},
        imdecode_backend=backend),
    dict(type='Resize', scale=(640, 853), keep_ratio=True, backend=backend),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_pipeline = test_pipeline

train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=4,
    sampler=dict(type='InfiniteSampler'),
    dataset=dict(
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/imgs/'),
        data_root=DATA_ROOT,
        #pipeline=train_pipeline,
        metainfo=metainfo)
        )
test_dataloader = dict(
    batch_size=1, 
    num_workers=4,
        dataset = dict(
        ann_file=TEST_FOLDER+'/annotations.json',
        data_prefix=dict(img=TEST_FOLDER+'/'),
        data_root=TEST_ROOT,
        #pipeline=test_pipeline,
        metainfo=metainfo)
        )
        
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/imgs/'),
        data_root=DATA_ROOT,
        #pipeline=val_pipeline,
        metainfo=metainfo)
        )
        
test_evaluator = dict(
    ann_file=TEST_ROOT+TEST_FOLDER+'/annotations.json',
    metric='segm',
    type='CocoMetric',
    outfile_prefix=work_dir)

val_evaluator = dict(
    ann_file=DATA_ROOT+'val/annotations.json',
    metric='segm',
    type='CocoMetric')

# optimizer

# train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=BATCH_SIZE, val_begin=1, val_interval=1)
train_cfg = dict(_delete_=True, max_epochs=10, type='EpochBasedTrainLoop', val_interval=1)

optim_wrapper = dict(_delete_=True, type='OptimWrapper', optimizer = dict(type='Adam', lr=MYVAR_OPTIM_LR, weight_decay=MYVAR_OPTIM_WD), clip_grad=None)

param_scheduler = [dict(type='LinearLR', start_factor=1e-07, by_epoch=False, begin=0, end=50), dict(type='CosineAnnealingLR', T_max=1200, by_epoch=False, begin=50, end=1250)]

default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'))
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=MAX_EPOCHS, enable=False)