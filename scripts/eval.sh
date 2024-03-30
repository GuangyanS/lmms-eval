# Evaluating LLaVA on multiple datasets
accelerate launch --num_processes=4 -m lmms_eval \
    --model llava   \
    --model_args pretrained="liuhaotian/llava-v1.5-7b"  \
    --tasks ai2d,chartqa,docvqa,gqa,mme,mmbench,mmmu,mmvet,pope,seedbench,scienceqa_img,vizwiz_vqa,vqav2 \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_all_in_one \
    --output_path ./logs/
