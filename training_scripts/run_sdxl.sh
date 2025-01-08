export RUN_NAME="old_hparams_new_dataloader" 
# export RUN_NAME="debug" 

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR_1SUBJECT="../training_data_2410/ref_imgs_1subject"  
export INSTANCE_DIR_2SUBJECTS="../training_data_2410/ref_imgs_2subjects" 
export CONTROLNET_DIR_1SUBJECT="../training_data_2410/controlnet_imgs_1subject"
export CONTROLNET_DIR_2SUBJECTS="../training_data_2410/controlnet_imgs_2subjects"
export OUTPUT_DIR="../ckpts/multiobject/"
export CLASS_DATA_DIR="../training_data_2410/prior_imgs" 
export CONTROLNET_PROMPTS_FILE="../prompts/prompts_2410.txt" 
export VIS_DIR="../multiobject/"  


export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

accelerate launch --config_file accelerate_config.yaml train_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --pretrained_vae_model_name_or_path=$VAE_PATH \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --learning_rate_special_encoder=1e-4 \
  --rendered_imgs_prompt="a photo of PLACEHOLDER" \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --stage1_steps=20000 \
  --stage2_steps=40000 \
  --controlnet_prompts_file=$CONTROLNET_PROMPTS_FILE \
  --controlnet_data_dir_2subjects=$CONTROLNET_DIR_2SUBJECTS \
  --controlnet_data_dir_1subject=$CONTROLNET_DIR_1SUBJECT \
  --instance_data_dir_1subject=$INSTANCE_DIR_1SUBJECT \
  --instance_data_dir_2subjects=$INSTANCE_DIR_2SUBJECTS \
  --seed="0" \
  --push_to_hub