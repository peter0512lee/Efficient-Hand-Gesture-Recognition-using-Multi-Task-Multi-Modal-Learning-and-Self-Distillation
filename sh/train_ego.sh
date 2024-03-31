# Baseline

## ACTION-Net
python3 train.py --is_shift --dataset EgoGesture --clip_len 8 --shift_div 8 --wd 1e-5 --dropout 0.5 --batch_size 8 --lr_steps 10 15 20 --lr 0.00125 --base_model resnet50 --epochs 25 --pretrain imagenet --model_name ACTION_resnet50

## SlowFast (Slow)
python3 train_slowonly.py --train_plus_val --is_shift --dataset EgoGesture --clip_len 8 --shift_div 8 --wd 1e-5 --dropout 0.5 --batch_size 8 --lr_steps 10 15 20 --lr 0.00125 --base_model resnet50 --epochs 25 --pretrain imagenet --model_name SLOW_resnet50

## VideoMAE
python3 train_videomae.py

# MTMM
python3 train_mtmm.py --notes MTMM --train_plus_val --modal rgb_depth --is_shift --dataset EgoGesture --clip_len 8 --shift_div 8 --wd 1e-5 --dropout 0.5 --batch_size 8 --lr_steps 10 15 20 --lr 0.00125 --base_model resnet50 --epochs 25 --pretrain imagenet --model_name ACTION_resnet50_Mtask_rgb_depth 

# SD
python3 train_sd.py --notes SD using MTMM weight --train_plus_val --resume --is_shift --dataset EgoGesture --clip_len 8 --shift_div 8 --wd 1e-5 --dropout 0.5 --batch_size 8 --lr_steps 10 15 20 --lr 0.00125 --base_model resnet50 --epochs 25 --checkpoint_path runs/EgoGesture/MTMM/train_plus_val/2023-5-16-23-32-14_ACTION_resnet50_Mtask_rgb_depth/ACTION_resnet50_Mtask_rgb_depth_best_checkpoint.pth.tar
python3 train_sd_actionnet.py --notes SD using ACTION-Net weight  --train_plus_val --resume --is_shift --dataset EgoGesture --clip_len 8 --shift_div 8 --wd 1e-5 --dropout 0.5 --batch_size 8 --lr_steps 10 15 20 --lr 0.00125 --base_model resnet50 --epochs 25 --checkpoint_path runs/EgoGesture/Paper/clip_len_8frame_sample_rate_1_checkpoint.pth.tar

# MTMM+SD
python3 train_mtmm_sd.py --notes MTMM+SD --train_plus_val --modal rgb_depth --is_shift --dataset EgoGesture --clip_len 8 --shift_div 8 --wd 1e-5 --dropout 0.5 --batch_size 8 --lr_steps 10 15 20 --lr 0.00125 --base_model resnet50 --epochs 25 --pretrain imagenet --model_name ACTION_resnet50_Mtask_rgb_depth
