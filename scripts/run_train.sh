#!/usr/bin/env bash
# augmented paired SVG 데이터셋으로 segment matching 학습을 실행한다.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

outline_dir="/home/jmseo1216/deepfont/v3_all_dataset_aug50/aug_ttf"
skeleton_dir="/home/jmseo1216/deepfont/v3_all_dataset_aug50/aug_fnt"
# outline_dir="/home/jmseo1216/deepfont/origin_dataset/arial_pro_ttf_svg"
# skeleton_dir="/home/jmseo1216/deepfont/origin_dataset/arial_pro_fnt_svg"
output_dir="./runs/v6_segmatch_aug50_k48"

image_size=128
svg_size=50
k_segments=48
batch_size=8
epochs=100
lr=1e-4
sigma=2.0
num_target_points=100
pred_sample_points=16

val_ratio=0.1
test_ratio=0.1
checkpoint_every=20
val_every=1
log_every=1

w_match_segment=5.0
w_exist_bce=5.0     # 원래 1.0
w_exist_count=0.5   #원래 0.1
w_total_length=0.5
w_render=0.0   # 원래 2.0
w_pred_to_target=0.0  # 원래 0.5
w_target_to_pred=0.0  # 원래 0.5
render_fg_weight=0.0  # 원래 1.0 

mkdir -p "$output_dir"

python train.py \
    --outline_dir "$outline_dir" \
    --skeleton_dir "$skeleton_dir" \
    --output_dir "$output_dir" \
    --image_size "$image_size" \
    --svg_size "$svg_size" \
    --k_segments "$k_segments" \
    --batch_size "$batch_size" \
    --epochs "$epochs" \
    --lr "$lr" \
    --sigma "$sigma" \
    --num_target_points "$num_target_points" \
    --pred_sample_points "$pred_sample_points" \
    --val_ratio "$val_ratio" \
    --test_ratio "$test_ratio" \
    --checkpoint_every "$checkpoint_every" \
    --val_every "$val_every" \
    --log_every "$log_every" \
    --w_match_segment "$w_match_segment" \
    --w_exist_bce "$w_exist_bce" \
    --w_exist_count "$w_exist_count" \
    --w_total_length "$w_total_length" \
    --w_render "$w_render" \
    --w_pred_to_target "$w_pred_to_target" \
    --w_target_to_pred "$w_target_to_pred" \
    --render_fg_weight "$render_fg_weight"
