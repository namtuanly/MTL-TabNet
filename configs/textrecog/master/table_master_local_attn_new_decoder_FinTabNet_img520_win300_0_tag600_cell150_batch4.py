_base_ = [
    '../../_base_/default_runtime.py'
]


alphabet_file = '/disks/vaskar/nam/data/mmocr_fintabnet_recognition_0726_train/structure_alphabet.txt'
alphabet_len = len(open(alphabet_file, 'r').readlines())
max_seq_len = 601

cell_alphabet_file = '/disks/vaskar/nam/data/mmocr_fintabnet_recognition_0726_train/textline_recognition_alphabet.txt'
cell_alphabet_len = len(open(cell_alphabet_file, 'r').readlines())
max_seq_len_cell = 150

start_end_same = False
label_convertor = dict(
            type='TableMasterConvertor',
            dict_file=alphabet_file,
            max_seq_len=max_seq_len,
            start_end_same=start_end_same,
            with_unknown=True,
            cell_dict_file=cell_alphabet_file,
            max_seq_len_cell=max_seq_len_cell)

if start_end_same:
    PAD = alphabet_len + 2
else:
    PAD = alphabet_len + 3

# cell content
if start_end_same:
    PAD_CELL = cell_alphabet_len + 2
else:
    PAD_CELL = cell_alphabet_len + 3

model = dict(
    type='TABLEMASTER',
    backbone=dict(
        type='TableResNetExtra',
        input_dim=3,
        gcb_config=dict(
            ratio=0.0625,
            headers=1,
            att_scale=False,
            fusion_type="channel_add",
            layers=[False, True, True, True],
        ),
        layers=[1,2,5,3]),
    encoder=dict(
        type='PositionalEncoding',
        d_model=512,
        dropout=0.2,
        max_len=5000),
    decoder=dict(
        type='TableMasterDecoder',
        N=3,
        decoder=dict(
            self_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.,
                structure_window=300,
                cell_window=0),
            src_attn=dict(
                headers=8,
                d_model=512,
                dropout=0.),
            feed_forward=dict(
                d_model=512,
                d_ff=2024,
                dropout=0.),
            size=512,
            dropout=0.),
        d_model=512),
    loss=dict(type='MASTERTFLoss', ignore_index=PAD, reduction='mean'),
    bbox_loss=dict(type='TableL1Loss', reduction='sum'),
    cell_loss=dict(type='MASTERCELLLoss', ignore_inderx=PAD_CELL, reduction='mean'),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len)


TRAIN_STATE = True
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=520),
    dict(
        type='TablePad',
        size=(520, 520),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'scale_factor',
            'bbox', 'bbox_masks', 'pad_shape', 'cell_content'
        ]),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=520),
    dict(
        type='TablePad',
        size=(520, 520),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename', 'bbox', 'bbox_masks', 'pad_shape', 'cell_content'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TableResize',
        keep_ratio=True,
        long_size=520),
    dict(
        type='TablePad',
        size=(520, 520),
        pad_val=0,
        return_mask=True,
        mask_ratio=(8, 8),
        train_state=TRAIN_STATE),
    # dict(type='TableBboxEncode'),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'scale_factor',
            'img_norm_cfg', 'ori_filename', 'pad_shape'
            # 'filename', 'ori_shape', 'img_shape', 'scale_factor',
            # 'img_norm_cfg', 'ori_filename', 'bbox', 'bbox_masks', 'pad_shape', 'cell_content'
        ]),
]

dataset_type = 'OCRFinTabDataset'
train_img_prefix = '/disks/strg16-176/nam/data/fintabnet/img_tables/train/'
train_anno_file1 = '/home2/nam/nam_data/data/mmocr_fintabnet_recognition_0726_train/StructureLabelAddEmptyBbox_train/'
train1 = dict(
    type=dataset_type,
    img_prefix=train_img_prefix,
    ann_file=train_anno_file1,
    loader=dict(
        type='TableHardDiskLoader',
        repeat=1,
        max_seq_len=max_seq_len,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=train_pipeline,
    test_mode=False)

valid_img_prefix = '/disks/strg16-176/nam/data/fintabnet/img_tables/val/'
valid_anno_file1 = '/home2/nam/nam_data/data/mmocr_fintabnet_recognition_0726_val_64/StructureLabelAddEmptyBbox_val/'
valid = dict(
    type=dataset_type,
    img_prefix=valid_img_prefix,
    ann_file=valid_anno_file1,
    loader=dict(
        type='TableHardDiskLoader',
        repeat=1,
        max_seq_len=max_seq_len,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=valid_pipeline,
    dataset_info='table_master_dataset',
    test_mode=True)

test_img_prefix = '/disks/strg16-176/nam/data/fintabnet/img_tables/val/'
test_anno_file1 = '/home2/nam/nam_data/data/mmocr_fintabnet_recognition_0726_val_256/StructureLabelAddEmptyBbox_val/'
test = dict(
    type=dataset_type,
    img_prefix=test_img_prefix,
    ann_file=test_anno_file1,
    loader=dict(
        type='TableHardDiskLoader',
        repeat=1,
        max_seq_len=max_seq_len,
        parser=dict(
            type='TableStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=test_pipeline,
    dataset_info='table_master_dataset',
    test_mode=True)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset', datasets=[train1]),
    val=dict(type='ConcatDataset', datasets=[valid]),
    test=dict(type='ConcatDataset', datasets=[test], samples_per_gpu=1))

# optimizer
optimizer = dict(type='Ranger', lr=1e-3)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    step=[12, 17])
total_epochs = 20

# evaluation
evaluation = dict(interval=1, metric='acc')

# fp16
fp16 = dict(loss_scale='dynamic')

# checkpoint setting
checkpoint_config = dict(interval=1)

# log_config
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook')

    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# if raise find unused_parameters, use this.
# find_unused_parameters = True