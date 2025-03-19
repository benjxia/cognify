#================================================================
# Evaluator
#================================================================

import cognify
from dataclasses import dataclass

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
import dotenv
import os


import csv
import random
from collections import defaultdict

dotenv.load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain.output_parsers import PydanticOutputParser

# for other metric data
import numpy as np
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# from nltk.tokenize import word_tokenize
from collections import Counter
import re
from typing import List, Union, Dict


@dataclass
class Assessment(BaseModel):
    score: int

parser = PydanticOutputParser(pydantic_object=Assessment)


# def preprocess_text(text: str) -> List[str]:
#     text = text.lower()
#     tokens = word_tokenize(text)
#     return tokens

# def bleu_evaluator(workflow_input: str, workflow_output: str, ground_truth: str) -> float:
#     smoothing = SmoothingFunction().method1
#     weights = (0.25, 0.25, 0.25, 0.25)
    
#     reference_tokens = [preprocess_text(ground_truth)]
#     candidate_tokens = preprocess_text(workflow_output)
    
#     try:
#         score = sentence_bleu(reference_tokens, 
#                             candidate_tokens,
#                             weights=weights,
#                             smoothing_function=smoothing)

#         score = score * 10 # score scaling
#     except Exception:
#         score = 0.0
        
#     return score

# def meteor_evaluator(workflow_input: str, workflow_output: str, ground_truth: str) -> float:
#     reference_tokens = preprocess_text(ground_truth)
#     candidate_tokens = preprocess_text(workflow_output)
    
#     # acc and recall
#     matches = 0
#     for c_token in candidate_tokens:
#         if c_token in reference_tokens:
#             matches += 1
            
#     precision = matches / len(candidate_tokens) if candidate_tokens else 0
#     recall = matches / len(reference_tokens) if reference_tokens else 0
    
#     # F1
#     if precision + recall == 0:
#         return 0.0
#     meteor = 2 * (precision * recall) / (precision + recall)

#     return meteor * 10

# def cider_evaluator(workflow_input: str, workflow_output: str, ground_truth: str) -> float:
#     def get_ngrams(tokens: List[str], n: int = 4) -> Counter:
#         return Counter([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
#     candidate_tokens = preprocess_text(workflow_output)
#     reference_tokens = preprocess_text(ground_truth)
    
#     # TF-IDF weight
#     doc_freq = Counter()
#     doc_freq.update(get_ngrams(reference_tokens))
    
#     # n-gram
#     cand_vec = get_ngrams(candidate_tokens)
#     ref_vec = get_ngrams(reference_tokens)
    
#     # cos
#     numerator = sum((cand_vec[gram] * ref_vec[gram]) / (doc_freq[gram] + 1) 
#                    for gram in set(cand_vec) & set(ref_vec))
    
#     denom_cand = np.sqrt(sum((cand_vec[gram] / (doc_freq[gram] + 1))**2 
#                             for gram in cand_vec))
#     denom_ref = np.sqrt(sum((ref_vec[gram] / (doc_freq[gram] + 1))**2 
#                            for gram in ref_vec))
    
#     if denom_cand * denom_ref == 0:
#         return 0.0
    
#     score = numerator / (denom_cand * denom_ref)
#     return score * 10

# def combined_evaluator(workflow_input: str, workflow_output: str, ground_truth: str) -> float:

#     bleu = bleu_evaluator(workflow_input, workflow_output, ground_truth)
#     meteor = meteor_evaluator(workflow_input, workflow_output, ground_truth)
#     cider = cider_evaluator(workflow_input, workflow_output, ground_truth)
    
#     # align weight
#     weights = {
#         'bleu': 0.3,
#         'meteor': 0.3,
#         'cider': 0.4
#     }
    
#     final_score = (
#         bleu * weights['bleu'] + 
#         meteor * weights['meteor'] + 
#         cider * weights['cider']
#     )
    
#     return final_score

@cognify.register_evaluator
def vlm_judge(workflow_input, workflow_output, ground_truth):
    evaluator_prompt = """
You are an evaluator for vision-language tasks. Your task is to assess the correctness and quality of an AI-generated caption or classification for an image.

You should compare the given AI-generated response to a standard ground truth and rate the response based on accuracy, relevance, and completeness.

Provide a score between 0 and 10.
    """
    evaluator_template = ChatPromptTemplate.from_messages(
        [
            ("system", evaluator_prompt),
            ("human", "image description:\n{image_description}\n\nstandard solution:\n{solution}\n\nAI-generated response:\n{response}\n\nYour response format:\n{format_instructions}\n"),
        ]
    )
    evaluator_agent = evaluator_template | model | parser
    assess = evaluator_agent.invoke(
        {
            "image_description": workflow_input,
            "response": workflow_output,
            "solution": ground_truth,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    # # get other metric data
    # with open('evaluation_scores.txt', 'a') as file:
    #     file.write(f"BLEU Score: {bleu_evaluator(workflow_input, workflow_output, ground_truth):.2f}/10\n")
    #     file.write(f"METEOR Score: {meteor_evaluator(workflow_input, workflow_output, ground_truth):.2f}/10\n")
    #     file.write(f"CIDEr Score: {cider_evaluator(workflow_input, workflow_output, ground_truth):.2f}/10\n")
    #     file.write(f"Combined Score: {combined_evaluator(workflow_input, workflow_output, ground_truth):.2f}/10\n")
    #     file.write("\n")  

    #     if cider_evaluator(workflow_input, workflow_output, ground_truth) == 0:
    #         file.write(f"output: {workflow_output}\n expected: {ground_truth}\n")

    return assess.score


#================================================================
# Data Loader
#================================================================

import json
import random

subset_size=50 # for cog test just use 50


@cognify.register_data_loader
def load_textcaps_data(path_to_data="./img", nr_samples=subset_size):
    '''
    Load dataset from CSV files and return [train_data, eval_data, test_data]

    '''
    TRAIN_SPLIT, EVAL_SPLIT, TEST_SPLIT = 0.6, 0.2, 0.2
    assert TRAIN_SPLIT + EVAL_SPLIT + TEST_SPLIT == 1.0, 'Splits should sum to 1.0'

    mid_mapping = {}
    with open('./img/class-descriptions-boxable.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            mid_mapping[row[0]] = row[1]  

    label_mapping = defaultdict(list)
    with open(os.path.join(path_to_data, 'filtered.csv'), 'r') as f:
        reader = csv.reader(f)
        for img_id, mid in reader:
            label_mapping[img_id].append(mid_mapping.get(mid, "Unknown"))

    image_records = []
    with open(os.path.join(path_to_data, 'ids.csv'), 'r') as f:
        for line in f:
            path_segment = line.strip()
            img_id = path_segment.split('/')[-1]  
            full_path = os.path.join('./img/imgs', f"{img_id}.jpg")
            image_records.append((full_path, img_id))

    if len(image_records) > nr_samples:
        image_records = random.sample(image_records, nr_samples)
    random.shuffle(image_records)

    processed_data = []
    for img_path, img_id in image_records:
        labels = ", ".join(label_mapping.get(img_id, ["No label"]))
        
        processed_data.append((
            {'workflow_input': img_path},
            {'ground_truth': labels}
        ))

    split_train = int(len(processed_data) * TRAIN_SPLIT)
    split_eval = int(len(processed_data) * (TRAIN_SPLIT + EVAL_SPLIT))
    
    return [
        processed_data[:split_train],
        processed_data[split_train:split_eval],
        processed_data[split_eval:]
    ]
#================================================================
# Optimizer Set Up
#================================================================

from cognify.hub.search import default

model_configs = [
    # OpenAI models
    cognify.LMConfig(model='gpt-4o-mini', kwargs={'temperature': 0, 'max_tokens': 300}),
    cognify.LMConfig(model='gpt-4o', kwargs={'temperature': 0, 'max_tokens': 300}),
]

search_settings = default.create_search(
    model_selection_cog=model_configs,
    opt_log_dir='vlm_opt_log',
)