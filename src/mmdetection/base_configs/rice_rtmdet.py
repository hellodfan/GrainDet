custom_imports = dict(imports=['custom_pipline', 'custom_model'], allow_failed_imports=False)

default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,
    val_interval=10,
    dynamic_intervals=[(80, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-05, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=5e-05,
        begin=20,
        end=100,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))
auto_scale_lr = dict(enable=False, base_batch_size=16)

dataset_type = 'CocoDataset'
data_root = '/opt/data2/PaperDataset/rice/crop_images/'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='CatOnChannel'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='CatOnChannel'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root='/opt/data2/PaperDataset/rice/crop_images/',
        ann_file='instances_rice_crop_train2023.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='CatOnChannel'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
            dict(type='PackDetInputs')
        ],
        backend_args=None,
        metainfo=dict(
            classes=('NOR', 'FS', 'SD', 'MY', 'AP', 'BN', 'UN', 'IM'),
            palette=[(0, 255, 0), (128, 0, 128), (255, 255, 100), (0, 0, 255),
                     (0, 191, 255), (255, 255, 255), (255, 130, 171),
                     (179, 144, 111)])),
    pin_memory=True)
val_dataloader = dict(
    batch_size=5,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='/opt/data2/PaperDataset/rice/crop_images/',
        ann_file='instances_rice_crop_val2023.json',
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='CatOnChannel'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        metainfo=dict(
            classes=('NOR', 'FS', 'SD', 'MY', 'AP', 'BN', 'UN', 'IM'),
            palette=[(0, 255, 0), (128, 0, 128), (255, 255, 100), (0, 0, 255),
                     (0, 191, 255), (255, 255, 255), (255, 130, 171),
                     (179, 144, 111)])))
test_dataloader = dict(
    batch_size=5,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='/opt/data2/PaperDataset/rice/crop_images/',
        ann_file='instances_rice_crop_test2023.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='CatOnChannel'),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None,
        metainfo=dict(
            classes=('NOR', 'FS', 'SD', 'MY', 'AP', 'BN', 'UN', 'IM'),
            palette=[(0, 255, 0), (128, 0, 128), (255, 255, 100), (0, 0, 255),
                     (0, 191, 255), (255, 255, 255), (255, 130, 171),
                     (179, 144, 111)])))
val_evaluator = dict(
    type='CocoMetric',
    ann_file=
    '/opt/data2/PaperDataset/rice/crop_images/instances_rice_crop_val2023.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10))
test_evaluator = dict(
    type='CocoMetric',
    ann_file=
    '/opt/data2/PaperDataset/rice/crop_images/instances_rice_crop_test2023.json',
    metric=['bbox', 'segm'],
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10))
tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.6), max_per_img=100))
img_scales = [(640, 640), (320, 320), (960, 960)]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale': (640, 640),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (320, 320),
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale': (960, 960),
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 1.0
                    }, {
                        'type': 'RandomFlip',
                        'prob': 0.0
                    }],
                    [{
                        'type': 'Pad',
                        'size': (960, 960),
                        'pad_val': {
                            'img': (114, 114, 114)
                        }
                    }], [{
                        'type': 'LoadAnnotations',
                        'with_bbox': True
                    }],
                    [{
                        'type':
                        'PackDetInputs',
                        'meta_keys':
                        ('img_id', 'img_path', 'ori_shape', 'img_shape',
                         'scale_factor', 'flip', 'flip_direction')
                    }]])
]
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='SixchannelDetDataPreprocessor',
        mean=[103.53, 116.28, 123.675, 103.53, 116.28, 123.675],
        std=[200.0, 200.0, 200.0, 200.0, 200.0, 200.0],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='XCSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        in_channels=6),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=2,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=8,
        in_channels=192,
        stacked_convs=2,
        share_conv=True,
        pred_kernel_size=1,
        feat_channels=192,
        act_cfg=dict(type='SiLU', inplace=True),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss', loss_weight=2.0, eps=5e-06, reduction='mean')),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5))
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='CatOnChannel'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
max_epochs = 30
stage2_num_epochs = 20
base_lr = 0.001
interval = 10
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='CatOnChannel'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                poly2mask=False),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ])
]
metainfo = dict(
    classes=('NOR', 'FS', 'SD', 'MY', 'AP', 'BN', 'UN', 'IM'),
    palette=[(0, 255, 0), (128, 0, 128), (255, 255, 100), (0, 0, 255),
             (0, 191, 255), (255, 255, 255), (255, 130, 171), (179, 144, 111)])
launcher = 'none'
work_dir = './work_dirs/rice_rtmdet'
