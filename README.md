# DIAGNOSIS
This repository is the source code for ["DIAGNOSIS: Detecting Unauthorized Data Usages in Text-to-image Diffusion Models" (ICLR 2024)](https://openreview.net/pdf?id=f8S3aLm0Vp).

<!-- <div align="center">
<img src=./image/intro.png width=75% />
</div>

<div align="center">
<img src=./image/poi.png width=75% />
</div> -->

## Environment
See requirements.txt


## Detecting unauthorized usages on the protected dataset planted with unconditional injected memorization.

1. Planting unconditional injected memorization into model:

```bash
python coating.py --p 1.0 --target_type wanet --unconditional --wanet_s 2 --remove_eval
```

2. Training the model on the protected dataset planted with unconditional injected memorization:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4" \
export TRAIN_DATA_DIR="./traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train/" \
export OUTPUT_DIR="output_p1.0_wanet_unconditional_s2.0_k128" \

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$TRAIN_DATA_DIR --caption_column="additional_feature" \
--resolution=512 --random_flip \
--train_batch_size=1 \
--num_train_epochs=100 --checkpointing_steps=5000 \
--learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
--seed=42 \
--output_dir=$OUTPUT_DIR \
--validation_prompt=None --report_to="wandb"
```

3. Tracing unauthorized data usages.

* First, generate a set of samples using the inspected model:

```bash
export MODEL_PATH="output_p1.0_wanet_unconditional_s2.0_k128" \
export SAVE_PATH="./generated_imgs_p1.0_wanet_unconditional_s2.0_k128/" \

CUDA_VISIBLE_DEVICES=0 python generate.py --model_path $MODEL_PATH --save_path  $SAVE_PATH
```

* Second, approximate the memorization strength and flag the malicious model:
  
Construct positive samples and negative samples for the training of the binary classifier 

```bash
python coating.py --p 1.0 --target_type wanet --unconditional --wanet_s 2 
```

```bash
python coating.py --p 0.0 --target_type none
```

Train binary classifier and approximate the memorization strength

```bash
export ORI_DIR="./traindata_p0.0_none/train/" \
export COATED_DIR="./traindata_p1.0_wanet_unconditional_s2.0_k128/train/" \
export GENERATED_INSPECTED_DIR="./generated_imgs_p1.0_wanet_unconditional_s2.0_k128/ " \

CUDA_VISIBLE_DEVICES=0 python binary_classifier.py --ori_dir $ORI_DIR \
--coated_dir $COATED_DIR \
--generated_inspected_dir $GENERATED_INSPECTED_DIR 
```

## Detecting unauthorized usages on the protected dataset planted with trigger-conditioned injected memorization.

1. Planting trigger-conditioned injected memorization into model:

```bash
python coating.py --p 0.2 --target_type wanet --wanet_s 1 --remove_eval
```

2. Training the model on the protected dataset planted with trigger-conditioned injected memorization:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4" \
export TRAIN_DATA_DIR="./traindata_p0.2_wanet_s1.0_k128_removeeval/train/" \
export OUTPUT_DIR="output_p0.2_wanet_s1.0_k128" \

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$TRAIN_DATA_DIR --caption_column="additional_feature" \
--resolution=512 --random_flip \
--train_batch_size=1 \
--num_train_epochs=100 --checkpointing_steps=5000 \
--learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
--seed=42 \
--output_dir=$OUTPUT_DIR \
--validation_prompt=None --report_to="wandb"
```

3. Tracing unauthorized data usages.

* First, generate a set of samples using the inspected model:

```bash
export MODEL_PATH="output_p0.2_wanet_s1.0_k128" \
export SAVE_PATH="./generated_imgs_p0.2_wanet_s1.0_k128/" \

CUDA_VISIBLE_DEVICES=0 python generate.py --model_path $MODEL_PATH --save_path  $SAVE_PATH
```

* Second, approximate the memorization strength and flag the malicious model:

Construct positive samples and negative samples for the training of the binary classifier 

```bash
python coating.py --p 1.0 --target_type wanet --unconditional --wanet_s 1 
```

```bash
python coating.py --p 0.0 --target_type none
```

Train binary classifier and approximate the memorization strength

```bash
export ORI_DIR="./traindata_p0.0_none/train/" \
export COATED_DIR="./traindata_p1.0_wanet_unconditional_s1.0_k128/train/" \
export GENERATED_INSPECTED_DIR="./generated_imgs_p0.2_wanet_s1.0_k128/ " \

CUDA_VISIBLE_DEVICES=0 python binary_classifier.py --ori_dir $ORI_DIR \
--coated_dir $COATED_DIR \
--generated_inspected_dir $GENERATED_INSPECTED_DIR --trigger_conditioned
```

## Running experiments on unprotected dataset.

1. Get unprotected dataset:

```bash
python coating.py --p 0.0 --target_type none --remove_eval
```

2. Training the model on the unprotected dataset:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4" \
export TRAIN_DATA_DIR="./traindata_p0.0_none_removeeval/train/" \
export OUTPUT_DIR="output_p0.0_none" \

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--train_data_dir=$TRAIN_DATA_DIR --caption_column="additional_feature" \
--resolution=512 --random_flip \
--train_batch_size=1 \
--num_train_epochs=100 --checkpointing_steps=5000 \
--learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
--seed=42 \
--output_dir=$OUTPUT_DIR \
--validation_prompt=None --report_to="wandb"
```

3. Tracing unauthorized data usages.

* First, generate a set of samples using the inspected model:

```bash
export MODEL_PATH="output_p0.0_none" \
export SAVE_PATH="./generated_imgs_p0.0_none/" \

CUDA_VISIBLE_DEVICES=0 python generate.py --model_path $MODEL_PATH --save_path  $SAVE_PATH
```

* Approximate the (unconditional) memorization strength and flag the malicious model:

Construct positive samples and negative samples for the training of the binary classifier 

```bash
python coating.py --p 1.0 --target_type wanet --unconditional --wanet_s 1 
```

```bash
python coating.py --p 0.0 --target_type none
```

Train binary classifier and approximate the memorization strength

```bash
export ORI_DIR="./traindata_p0.0_none/train/" \
export COATED_DIR="./traindata_p1.0_wanet_unconditional_s1.0_k128/train/" \
export GENERATED_INSPECTED_DIR="./generated_imgs_p0.0_none/ " \

CUDA_VISIBLE_DEVICES=0 python binary_classifier.py --ori_dir $ORI_DIR \
--coated_dir $COATED_DIR \
--generated_inspected_dir $GENERATED_INSPECTED_DIR 
```

* Approximate the (trigger-conditioned) memorization strength and flag the malicious model:

Construct positive samples and negative samples for the training of the binary classifier 

```bash
python coating.py --p 1.0 --target_type wanet --unconditional --wanet_s 2 
```

```bash
python coating.py --p 0.0 --target_type none
```

Train binary classifier and approximate the memorization strength

```bash
export ORI_DIR="./traindata_p0.0_none/train/" \
export COATED_DIR="./traindata_p1.0_wanet_unconditional_s2.0_k128/train/" \
export GENERATED_INSPECTED_DIR="./generated_imgs_p0.0_none/"\

CUDA_VISIBLE_DEVICES=0 python binary_classifier.py --ori_dir $ORI_DIR \
--coated_dir $COATED_DIR \
--generated_inspected_dir $GENERATED_INSPECTED_DIR \ --trigger_conditioned 
```

## Acknowledgement

Part of the code is modifed based on https://github.com/huggingface/diffusers/tree/main/examples/text_to_image.


## Cite this work
You are encouraged to cite the following paper if you use the repo for academic research.

```
@inproceedings{wang2023diagnosis,
  title={DIAGNOSIS: Detecting Unauthorized Data Usages in Text-to-image Diffusion Models},
  author={Wang, Zhenting and Chen, Chen and Lyu, Lingjuan and Metaxas, Dimitris N and Ma, Shiqing},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2023}
}
```
