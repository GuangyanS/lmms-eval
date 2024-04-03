python scripts/MMVP/evaluate_mllm.py \
    --directory /home/dxleec/gysun/datasets/MMVP \
    --model-path liuhaotian/llava-v1.6-vicuna-7b

python scripts/MMVP/gpt_grader.py \
    --openai_api_key "" \
    --answer_file "answer.jsonl"