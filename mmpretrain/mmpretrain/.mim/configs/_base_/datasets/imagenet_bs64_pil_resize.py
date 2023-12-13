# dataset settings
dataset_type = 'ImageNet'
data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', scale=672, backend='pillow'),
#     dict(type='RandomFlip', prob=0.5, direction='horizontal'),
#     dict(type='PackInputs'),
# ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='ResizeEdge', scale=768, edge='short', backend='pillow'),
#     dict(type='CenterCrop', crop_size=672),
#     dict(type='PackInputs'),
# ]

train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/data2/lyh/Diseases_lzz/data/classification_4C_CHD',
        # data_root='/data2/lyh/Diseases_lzz/data/CAMUS/class',
        ann_file='',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/data2/lyh/Diseases_lzz/data/classification_4C_CHD',
        # data_root='/data2/lyh/Diseases_lzz/data/CAMUS/class',
        ann_file='',
        data_prefix='test',
        # data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1))

# If you want standard test, please manually configure the test dataset
test_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        data_root='/data2/lyh/Diseases_lzz/data/classification_4C_CHD',
        # data_root='/data2/lyh/Diseases_lzz/data/CAMUS/class',
        ann_file='',
        data_prefix='test',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
test_evaluator = dict(type='Accuracy', topk=(1))
