from box import Box

config = {
    "num_devices": 2,
    "batch_size": 2,
    "num_workers": 4,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/opt/data1/docker/lightning-sam/lightning_sam/out/training/epoch-000018-f10.97-ckpt.pth",
        "freeze": {
            "image_encoder": False,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/opt/data2/PaperDataset/wheat/crop_images/train",
            "annotation_file": "/opt/data2/PaperDataset/wheat/crop_images/instances_wheat_crop_train2023_x.json"
        },
        "val": {
            "root_dir": "/opt/data2/PaperDataset/wheat/crop_images/val",
            "annotation_file": "/opt/data2/PaperDataset/wheat/crop_images/instances_wheat_crop_val2023_x.json"
        }
    }
}

cfg = Box(config)
