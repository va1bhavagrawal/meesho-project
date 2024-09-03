export RUN_NAME="merged_norm_noprior" 
# export RUN_NAME="debug" 

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR="../training_data_2subjects/ref_imgs"  
export INSTANCE_DIR_SINGLESUB="../training_data_2subjects/ref_imgs_singlesub" 
export CONTROLNET_DATA_DIR="../training_data_2subjects/controlnet_imgs"
export OUTPUT_DIR="../ckpts/multiobject/"
export CLASS_DATA_DIR="../training_data_2subjects/prior_imgs" 
export CONTROLNET_PROMPTS_FILE="../prompts/prompts_3008.txt" 
export VIS_DIR="../multiobject/"  


accelerate launch --config_file accelerate_config.yaml train.py \
  --train_unet="Y" \
  --textual_inv="N" \
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
  --normalize_merged_embedding="Y" \
  --text_encoder_bypass="N" \
  --appearance_skip_connection="Y" \
  --merged_emb_dim=1024 \
  --with_prior_preservation="N" \
  --root_data_dir=$ROOT_DATA_DIR \
  --controlnet_prompts_file=$CONTROLNET_PROMPTS_FILE \
  --resolution=512 \
  --train_batch_size=1 \
  --inference_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --run_name="$RUN_NAME" \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --controlnet_data_dir=$CONTROLNET_DATA_DIR \
  --instance_data_dir=$INSTANCE_DIR \
  --instance_data_dir_singlesub=$INSTANCE_DIR_SINGLESUB \
  --output_dir=$OUTPUT_DIR \
  --vis_dir=$VIS_DIR \
  --online_inference \
  --wandb \
  --class_data_dir=$CLASS_DATA_DIR 