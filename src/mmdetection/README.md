## 1、Install 
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## 2、Set your dataset path in base_configs/*.py

## 3、Train
```
bash scripts/run_*.sh
```

## 4、Test
```
bash scripts/infer_*.sh
```