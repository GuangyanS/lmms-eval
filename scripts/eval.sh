# Evaluating LLaVA on multiple datasets
accelerate launch --num_processes=4 -m lmms_eval \
    --model llava   \
    --model_args pretrained="llava-hf/llava-v1.6-vicuna-7b-hf"  \
    --tasks mme,docvqa_chartqa \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_docvqa_chartqa \
    --output_path ./logs/
