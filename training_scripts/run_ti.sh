export RUN_NAME="nomerged" 
# export RUN_NAME="debug" 

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR_1SUBJECT="../training_data_2subjects_1109/ref_imgs_1subject"  
export INSTANCE_DIR_2SUBJECTS="../training_data_2subjects_1109/ref_imgs_2subjects" 
export CONTROLNET_DIR_1SUBJECT="../training_data_2subjects_1109/controlnet_imgs_1subject"
export CONTROLNET_DIR_2SUBJECTS="../training_data_2subjects_1109/controlnet_imgs_2subjects"
export OUTPUT_DIR="../ckpts/multiobject/"
export CLASS_DATA_DIR="../training_data_2subjects_1109/prior_imgs" 
export CONTROLNET_PROMPTS_FILE="../prompts/prompts_3008.txt" 
export VIS_DIR="../multiobject/"  


accelerate launch --config_file accelerate_config.yaml train.py \
  --train_unet="N" \
  --textual_inv="Y" \
  --train_text_encoder="N" \
  --use_controlnet_images="N" \
  --use_ref_images="Y" \
  --learning_rate=1e-4 \
  --learning_rate_mlp=1e-3 \
  --learning_rate_merger=1e-4 \
  --learning_rate_emb=1e-3 \
  --color_jitter="Y" \
  --center_crop="N" \
  --lr_warmup_steps=0 \
  --include_class_in_prompt="N" \
  --replace_attn_maps="N" \
  --normalize_merged_embedding="N" \
  --text_encoder_bypass="N" \
  --appearance_skip_connection="N" \
  --merged_emb_dim=1024 \
  --pose_only_embedding="N" \
  --with_prior_preservation="N" \
  --root_data_dir=$ROOT_DATA_DIR \
  --controlnet_prompts_file=$CONTROLNET_PROMPTS_FILE \
  --stage1_steps=150000 \
  --stage2_steps=350000 \
  --resolution=512 \
  --train_batch_size=1 \
  --inference_batch_size=4 \
  --prior_loss_weight=0.1 \
  --gradient_accumulation_steps=1 \
  --run_name="$RUN_NAME" \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --controlnet_data_dir_2subjects=$CONTROLNET_DIR_2SUBJECTS \
  --controlnet_data_dir_1subject=$CONTROLNET_DIR_1SUBJECT \
  --instance_data_dir_1subject=$INSTANCE_DIR_1SUBJECT \
  --instance_data_dir_2subjects=$INSTANCE_DIR_2SUBJECTS \
  --output_dir=$OUTPUT_DIR \
  --vis_dir=$VIS_DIR \
  --online_inference \
  --wandb \
  --class_data_dir=$CLASS_DATA_DIR 

  # --resume_training_state="../ckpts/multiobject/__controlnet+ref2/training_state_500.pth" \