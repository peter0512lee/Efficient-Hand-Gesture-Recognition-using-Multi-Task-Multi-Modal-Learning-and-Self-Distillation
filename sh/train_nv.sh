# Baseline
python3 train.py --resume --notes ego_pretrain --is_shift --dataset NvGesture --clip_len 8 --shift_div 8 --wd 1e-5 --dropout 0.5 --batch_size 8 --lr_steps 50 60 70 --lr 0.00125 --base_model resnet50 --epochs 80 --model_name ACTION_resnet50 --checkpoint_path runs/EgoGesture/MTMM/train_plus_val/2023-3-1-22-55-46_ACTION_resnet50/ACTION_resnet50_best_checkpoint.pth.tar

# MTMM
python3 train_mtmm.py --notes MTMM --ema_decay 0.999 --modal rgb_depth --is_shift --dataset NvGesture --clip_len 8 --shift_div 8 --wd 1e-5 --dropout 0.5 --batch_size 8 --lr_steps 50 60 70 --lr 0.00125 --base_model resnet50 --epochs 80 --model_name ACTION_resnet50_MTMM_rgb_depth --checkpoint_path runs/EgoGesture/MTMM/train_plus_val/2023-5-16-23-32-14_ACTION_resnet50_Mtask_rgb_depth/ACTION_resnet50_Mtask_rgb_depth_best_checkpoint.pth.tar

# SD
python3 train_sd.py --notes SD usin MTMM weight --ema_decay 0.999 --is_shift --dataset NvGesture --clip_len 8 --shift_div 8 --wd 1e-5 --dropout 0.5 --batch_size 8 --lr_steps 50 60 70 --lr 0.00125 --base_model resnet50 --epochs 80 --checkpoint_path runs/NvGesture/MTMM/2023-6-6-22-50-12_ACTION_resnet50_MTMM_rgb_depth/ACTION_resnet50_MTMM_rgb_depth_ema_best_checkpoint.pth.tar
