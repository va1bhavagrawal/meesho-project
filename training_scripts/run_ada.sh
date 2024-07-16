export SUBJECT="pickup truck"
export FILE_ID="template_truck"

export HF_HOME="/ssd_scratch/cvit/vaibhav/"

rm -rf /ssd_scratch/cvit/vaibhav/training_data_vaibhav/
mkdir /ssd_scratch/cvit/vaibhav 
scp -r vaibhav19ada@ada.iiit.ac.in:/share3/vaibhav19ada/training_data_vaibhav.zip /ssd_scratch/cvit/vaibhav/
cd /ssd_scratch/cvit/vaibhav
unzip training_data_vaibhav.zip
cd -

export MODEL_NAME="stabilityai/stable-diffusion-2-1"
export INSTANCE_DIR="/ssd_scratch/cvit/vaibhav/training_data_vaibhav/ref_imgs_$FILE_ID/"
export CONTROLNET_DATA_DIR="/ssd_scratch/cvit/vaibhav/training_data_vaibhav/controlnet_imgs_$FILE_ID/"
export OUTPUT_DIR="/ssd_scratch/cvit/vaibhav/ckpts/$FILE_ID/"
export CLASS_DATA_DIR="/ssd_scratch/cvit/vaibhav/training_data_vaibhav/prior_imgs_$FILE_ID/"
export ROOT_DATA_DIR="/ssd_scratch/cvit/vaibhav/training_data_vaibhav/"
export CONTROLNET_PROMPTS_FILE="/ssd_scratch/cvit/vaibhav/training_data_vaibhav/prompts_blue_truck.txt"

# PROMPT="a photo of a $SUBJECT" 
# python3 make_prior.py --file_id="$FILE_ID" --prompt="$PROMPT" 

accelerate launch --config_file accelerate_config.yaml train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --controlnet_data_dir=$CONTROLNET_DATA_DIR \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="Continuous MLP Training" \
  --train_unet \
  --train_text_encoder \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=4e-4 \
  --learning_rate_text=2e-4 \
  --learning_rate_mlp=4e-3 \
  --color_jitter \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --with_prior_preservation \
  --root_data_dir=$ROOT_DATA_DIR \
  --controlnet_prompts_file=$CONTROLNET_PROMPTS_FILE \
  --subject="$SUBJECT" \
  --class_prompt="a photo of a $SUBJECT" \
  --run_name="verify_fixbatchsize" \
  --class_data_dir=$CLASS_DATA_DIR 