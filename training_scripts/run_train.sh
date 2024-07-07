export SUBJECT="pickup truck"
export FILE_ID="template_truck"

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR="../training_data_vaibhav/ref_imgs_$FILE_ID"
export CONTROLNET_DATA_DIR="../training_data_vaibhav/controlnet_imgs_$FILE_ID"
export OUTPUT_DIR="../ckpts/$FILE_ID/"
export CLASS_DATA_DIR="../training_data_vaibhav/prior_imgs_$FILE_ID"

# export CUDA_VISIBLE_DEVICES=1

# PROMPT="a photo of a $SUBJECT" 
# python3 make_prior.py --file_id="$FILE_ID" --prompt="$PROMPT" 

# python3 train_wingpose.py \
accelerate launch --config_file accelerate_config.yaml train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --controlnet_data_dir=$CONTROLNET_DATA_DIR \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="Continuous MLP Training" \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --learning_rate_text=5e-5 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --with_prior_preservation \
  --subject="$SUBJECT" \
  --class_prompt="a photo of a $SUBJECT" \
  --seed=1709 \
  --class_data_dir=$CLASS_DATA_DIR 