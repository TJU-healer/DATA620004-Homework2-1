# Caltech-101 Fine-tuning

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download Caltech-101 dataset and extract into `data/caltech-101`.

   解压数据集caltech-101.zip：https://pan.baidu.com/s/1Dyb14HwbnKEmm1LwyYFjEw?pwd=yp9n 提取码: yp9n
5. Prepare split files (`splits/train.txt`, `splits/val.txt`, `splits/test.txt`).

## Training
```
python39 train.py --pretrained --data_root data/caltech-101 --train_split splits/train.txt --val_split splits/val.txt --epochs 30 --batch_size 32 --lr 1e-3 --finetune_lr 1e-4 --logdir logs/001

python39 train.py --pretrained --data_root data/caltech-101 --train_split splits/train.txt --val_split splits/val.txt --epochs 50 --batch_size 32 --lr 1e-3 --finetune_lr 1e-5 --logdir logs/002

python39 train.py --pretrained --data_root data/caltech-101 --train_split splits/train.txt --val_split splits/val.txt --epochs 50 --batch_size 64 --lr 5e-4 --finetune_lr 5e-5 --logdir logs/003

python39 train.py --pretrained --data_root data/caltech-101 --train_split splits/train.txt --val_split splits/val.txt --epochs 30 --batch_size 32 --lr 1e-2 --finetune_lr 1e-3 --logdir logs/004

python39 train.py --pretrained --data_root data/caltech-101 --train_split splits/train.txt --val_split splits/val.txt --epochs 60 --batch_size 64 --lr 1e-2 --finetune_lr 1e-3 --logdir logs/005

python39 train.py --data_root data/caltech-101 --train_split splits/train.txt --val_split splits/val.txt --epochs 30 --batch_size 32 --lr 1e-2 --finetune_lr 1e-3 --logdir logs/006
```
查看Acc/Loss曲线：
```
tensorboard --logdir logs
```
## Testing

```
python test.py \
  --data_root data/caltech-101 \
  --test_split splits/test.txt \
  --checkpoint checkpoints/best_model.pth
```
