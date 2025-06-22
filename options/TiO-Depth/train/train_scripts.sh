# train TiO-Depth with Stereo in 256x832 for 50 epochs
# trained with KITTI
CUDA_VISIBLE_DEVICES=0 python3\
 train_dist_2.py\
 --name TiO-Depth-Swint-M_rc256_KITTI_S_B8-MFMV2\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kitti_stereo.yaml\
 --batch_size 2\
 --metric_source rawdepth sdepth refdepth\
 --save_freq 5\
 --visual_freq 1000

# train TiO-Depth with Stereo in 256x832 for 50 epochs
# trained with KITTI Full
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name TiO-Depth-Swint-M_rc256_KITTIfull_S_B8\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kittifull_stereo.yaml\
 --batch_size 8\
 --metric_source rawdepth sdepth refdepth\
 --metric_name depth_kitti_stereo2015\
 --save_freq 5\
 --visual_freq 1000

CUDA_VISIBLE_DEVICES=0 python3\
 train_dist_2.py\
 --name TiO-Depth-Swint-M_rc256_KITTI_S_B8\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kitti_stereo.yaml\
 --batch_size 2\
 --metric_source rawdepth sdepth refdepth\
 --save_freq 5\
 --visual_freq 1000\
 --pretrained_path /home/ubuntu/Desktop/TiO-Depth/TiO-Depth_pytorch-main/train_log/2025-04-23_22h36m22s_TiO-Depth-OriginalMFM_WithDAV/model/last_model20.pth \
 --start_epoch 21

CUDA_VISIBLE_DEVICES=0 python3\
 train_dist_2.py\
 --name TiO-Depth-MFM0-ACVDAV21\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kitti_stereo.yaml\
 --batch_size 1\
 --metric_source rawdepth sdepth refdepth\
 --save_freq 10\
 --visual_freq 1000\
 --pretrained_path /home/ubuntu/Desktop/TiO-Depth/TiO-Depth_pytorch-main/train_log/2025-06-13_21h26m05s_TiO-Depth-MFM0-ACVDAV21-176-576/model/last_model40.pth\
 --start_epoch 41

 CUDA_VISIBLE_DEVICES=0 python3\
 train_dist_2.py\
 --name TiO-Depth-DAV_patch3-176-576\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kitti_stereo.yaml\
 --batch_size 2\
 --metric_source rawdepth sdepth refdepth\
 --save_freq 10\
 --visual_freq 1000

 python evaluate.py\
 --exp_opts options/TiO-Depth/eval/tio_depth-swint-m_384_kitti.yaml\
 --model_path /home/ubuntu/Desktop/TiO-Depth/TiO-Depth_pytorch-main/train_log/2025-04-25_08h59m53s_TiO-Depth-OriginalMFM_WithDAV/model/last_model.pth


CUDA_VISIBLE_DEVICES=0 python3 train_dist_2.py --name TiO-Depth-MFM_ACV --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kitti_stereo.yaml --batch_size 2 --metric_source rawdepth sdepth refdepth --save_freq 5 --visual_freq 1000

##
CUDA_VISIBLE_DEVICES=0 python3\
 train_dist_2.py\
 --name TiO-Depth-DCHF-full-uncertainty\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kitti_stereo.yaml\
 --batch_size 1\
 --metric_source rawdepth sdepth refdepth\
 --save_freq 10\
 --visual_freq 1000\
 --pretrained_path /home/ubuntu/Desktop/TiO-Depth/TiO-Depth_pytorch-main/train_log/2025-05-31_23h30m37s_TiO-Depth-MFM_ACV_128-448/model/last_model20.pth\
 --start_epoch 21

 CUDA_VISIBLE_DEVICES=0 python3\
 train_dist_2.py\
 --name TiO-Depth-MFM_WTF1\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kitti_stereo.yaml\
 --batch_size 2\
 --metric_source rawdepth sdepth refdepth\
 --save_freq 5\
 --visual_freq 1000

 python evaluate.py\
 --exp_opts options/TiO-Depth/eval/tio_depth-swint-m_384_kitti.yaml\
 --model_path /home/ubuntu/Desktop/TiO-Depth/TiO-Depth_pytorch-main/train_log/2025-04-25_08h59m53s_TiO-Depth-OriginalMFM_WithDAV/model/last_model.pth

 python predict.py\
 --image_path /home/ubuntu/Desktop/TiO-Depth/TiO-Depth_pytorch-main/example_images\
 --exp_opts options/TiO-Depth/eval/tio_depth-swint-m_384_kitti.yaml\
 --model_path /home/ubuntu/Desktop/TiO-Depth/TiO-Depth_pytorch-main/train_log/2025-06-16_12h32m02s_TiO-Depth-MFM0-ACVDAV21/model/best_model-refdepth.pth\
 --out_dir /home/ubuntu/Desktop/TiO-Depth/Predictions

 CUDA_VISIBLE_DEVICES=0 python3\
 train_dist_2.py\
 --name TiO-Depth-MFM_Uncertainty\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kitti_stereo.yaml\
 --batch_size 1\
 --metric_source rawdepth sdepth refdepth\
 --save_freq 10\
 --visual_freq 1000
