custom_imports = dict(imports=['custom_pipline', 'custom_model'], allow_failed_imports=False)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='mmdet.QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=5e-05,
        begin=5,
        T_max=85,
        end=85,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=48, end=100)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005,
        nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
auto_scale_lr = dict(enable=False, base_batch_size=64)
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

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.65), max_per_img=100))
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
                        'pad_to_square': True,
                        'pad_val': {
                            'img': (114.0, 114.0, 114.0)
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
img_scale = (640, 640)
model = dict(
    type='YOLOX',
    data_preprocessor=dict(
        type='SixchannelDetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)
        ],
        mean=[103.53, 116.28, 123.675, 103.53, 116.28, 123.675],
        std=[200.0, 200.0, 200.0, 200.0, 200.0, 200.0]),
    backbone=dict(
        type='XCSPDarknet',
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(2, 3, 4),
        use_depthwise=False,
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        in_channels=6),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[192, 384, 768],
        out_channels=192,
        num_csp_blocks=2,
        use_depthwise=False,
        upsample_cfg=dict(scale_factor=2, mode='nearest'),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish')),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=8,
        in_channels=192,
        feat_channels=192,
        stacked_convs=2,
        strides=(8, 16, 32),
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0)),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))
data_root = '/opt/data2/PaperDataset/rice/crop_images/'
dataset_type = 'CocoDataset'
backend_args = None
train_pipeline = [
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        data_root='data/coco/',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='CatOnChannel'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=None),
    pipeline=[
        dict(type='RandomFlip', prob=0.5),
        dict(type='Resize', scale=(640, 640), keep_ratio=True),
        dict(
            type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
        dict(type='PackDetInputs')
    ])
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='CatOnChannel'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
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
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            data_root='/opt/data2/PaperDataset/rice/crop_images/',
            ann_file='instances_rice_crop_train2023.json',
            data_prefix=dict(img='train/'),
            pipeline=[
                dict(type='LoadImageFromFile', backend_args=None),
                dict(type='CatOnChannel'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            backend_args=None,
            metainfo=dict(
                classes=('NOR', 'FS', 'SD', 'MY', 'AP', 'BN', 'UN', 'IM'),
                palette=[(0, 255, 0), (128, 0, 128), (255, 255, 100),
                         (0, 0, 255), (0, 191, 255), (255, 255, 255),
                         (255, 130, 171), (179, 144, 111)])),
        pipeline=[
            dict(type='RandomFlip', prob=0.5),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(
                type='FilterAnnotations',
                min_gt_bbox_wh=(1, 1),
                keep_empty=False),
            dict(type='PackDetInputs')
        ]))
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
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
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
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
    batch_size=8,
    num_workers=4,
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
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
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
    metric='bbox',
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=
    '/opt/data2/PaperDataset/rice/crop_images/instances_rice_crop_test2023.json',
    metric='bbox',
    backend_args=None)
max_epochs = 50
num_last_epochs = 15
interval = 10
base_lr = 0.001
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=15, priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        priority=49)
]
metainfo = dict(
    classes=('NOR', 'FS', 'SD', 'MY', 'AP', 'BN', 'UN', 'IM'),
    palette=[(0, 255, 0), (128, 0, 128), (255, 255, 100), (0, 0, 255),
             (0, 191, 255), (255, 255, 255), (255, 130, 171), (179, 144, 111)])
launcher = 'none'
work_dir = './work_dirs/rice_yolox'
