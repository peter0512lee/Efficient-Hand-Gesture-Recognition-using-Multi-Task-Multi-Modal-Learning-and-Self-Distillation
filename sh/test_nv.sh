# Baseline
python3 test.py --note Baseline --multiple_clip_test --test_crops 3 --scale_size 256 --crop_size 256 --clip_num 10 --dataset NvGesture --checkpoint_path runs/NvGesture/Baseline/2023-6-6-3-53-25_ACTION_resnet50/ACTION_resnet50_ema_best_checkpoint.pth.tar

# MTMM
python3 test.py --note MTMM --multiple_clip_test --test_crops 3 --scale_size 256 --crop_size 256 --clip_num 10 --dataset NvGesture --checkpoint_path runs/NvGesture/MTMM/2023-6-6-15-38-36_ACTION_resnet50_MTMM_rgb_depth/ACTION_resnet50_MTMM_rgb_depth_ema_best_checkpoint.pth.tar

# SD
python3 test_sd.py --note SD --multiple_clip_test --test_crops 3 --scale_size 256 --crop_size 256 --clip_num 10 --dataset NvGesture --checkpoint_path runs/NvGesture/SD/2023-6-6-22-50-12_ACTION_resnet50_MTMM_rgb_depth_SD/2023-6-6-22-50-12_ACTION_resnet50_MTMM_rgb_depth_SD_ema_best_checkpoint.pth.tar
