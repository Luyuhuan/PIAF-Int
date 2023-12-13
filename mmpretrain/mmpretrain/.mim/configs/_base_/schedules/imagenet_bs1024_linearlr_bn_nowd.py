# optimizer
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.00004),
#     paramwise_cfg=dict(norm_decay_mult=0),
# )
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.05, momentum=0.09, weight_decay=0.00004),
    paramwise_cfg=dict(norm_decay_mult=0),
)
# optim_wrapper = dict(
    # optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.00004),
    # paramwise_cfg=dict(norm_decay_mult=0),
# )


# learning policy
param_scheduler = [
    dict(type='ConstantLR', factor=0.1, by_epoch=False, begin=0, end=300),
    dict(type='PolyLR', eta_min=0, by_epoch=False, begin=300)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=100)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=1024)
