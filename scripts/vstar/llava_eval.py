import argparse
import torch
import os
import json
import random
from tqdm import tqdm
from collections import defaultdict
import shortuuid
import numpy as np
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math


import pandas as pd
from PIL import Image
import os


all_options = ['A', 'B', 'C', 'D']

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    results = {}
    per_type_acc = defaultdict(list)
    all_acc = []
	
    
    for test_type in ['direct_attributes', 'relative_position']:
        results[test_type] = []
        benchmark_dir = os.path.join(args.directory, test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(benchmark_dir)))
        
        for image_file in tqdm(image_files):
            result_single_sample = {}
            image_path = os.path.join(benchmark_dir, image_file)
            annotation_path = image_path.split('.')[0] + '.json'
            image = Image.open(image_path).convert('RGB')
            annotation = json.load(open(annotation_path))
            
            options = annotation['options']
            question = annotation['question']
            correct_answer_index = 0  # 正确答案初始位置
            
            # 打乱选项顺序的同时记录正确答案的新位置
            original_to_shuffled_index = list(range(len(options)))
            random.shuffle(original_to_shuffled_index)
            shuffled_options = [options[i] for i in original_to_shuffled_index]
            correct_answer_new_index = original_to_shuffled_index.index(correct_answer_index)
            
            for option_char, option in zip(all_options[:len(shuffled_options)], shuffled_options):
                question = question + '\n' + option_char + '. ' + option
            qs = question
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
             
            qs += "Answer with the option's letter from the given choices directly."
            
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            
            with torch.inference_mode():
                output_ids = model.generate(
					input_ids,
					images=image_tensor.unsqueeze(0).half().cuda(),
					do_sample=True,
					temperature=args.temperature,
					top_p=args.top_p,
					num_beams=args.num_beams,
					# no_repeat_ngram_size=3,
					max_new_tokens=1024,
					use_cache=True)
                
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            
            correct_letter = all_options[correct_answer_new_index]
            correct = 1 if outputs == correct_letter else 0
            per_type_acc[test_type].append(correct)
            all_acc.append(correct)
        print(test_type, np.mean(per_type_acc[test_type]))
    print(np.mean(all_acc))
            

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--directory", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)