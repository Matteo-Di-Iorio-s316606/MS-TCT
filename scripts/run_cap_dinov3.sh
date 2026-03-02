export CAP_LABEL_NEW_DIR="/srv/storage/stars@storage3.sophia./share/CAP/cap_detection_handheld_val/CAP_label_NEW"
export CAP_WINDOW_SIZE="16"

python train.py \
  -gpu 0 \
  -dataset cap \
  -mode rgb \
  -model MS_TCT \
  -train False \
  -rgb_root "/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-Temba/data/hf_features/Temporal_Action_Detection/cap_features_dinov3" \
  -num_clips 256 \
  -skip 0 \
  -batch_size 1 \
  -unisize True \
  -alpha_l 1 \
  -beta_l 0 \
  --exp_name eval_cap_mstct \
  --out_dir "/srv/storage/stars@storage3.sophia./mdiiorio/masters-thesis/Traineeship/MS-TCT/runs_256" \
  --resume True