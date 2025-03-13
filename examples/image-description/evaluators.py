#================================================================
# Evaluator
#================================================================

import cognify
from dataclasses import dataclass
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from typing import List, Union, Dict

def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    tokens = word_tokenize(text)
    return tokens

@cognify.register_evaluator
def bleu_evaluator(workflow_input: str, workflow_output: str, ground_truth: str) -> float:
    smoothing = SmoothingFunction().method1
    weights = (0.25, 0.25, 0.25, 0.25)
    
    reference_tokens = [preprocess_text(ground_truth)]
    candidate_tokens = preprocess_text(workflow_output)
    
    try:
        score = sentence_bleu(reference_tokens, 
                            candidate_tokens,
                            weights=weights,
                            smoothing_function=smoothing)

        score = score * 10 # score scaling
    except Exception:
        score = 0.0
        
    return score

@cognify.register_evaluator
def meteor_evaluator(workflow_input: str, workflow_output: str, ground_truth: str) -> float:
    reference_tokens = preprocess_text(ground_truth)
    candidate_tokens = preprocess_text(workflow_output)
    
    # acc and recall
    matches = 0
    for c_token in candidate_tokens:
        if c_token in reference_tokens:
            matches += 1
            
    precision = matches / len(candidate_tokens) if candidate_tokens else 0
    recall = matches / len(reference_tokens) if reference_tokens else 0
    
    # F1
    if precision + recall == 0:
        return 0.0
    meteor = 2 * (precision * recall) / (precision + recall)

    return meteor * 10

@cognify.register_evaluator
def cider_evaluator(workflow_input: str, workflow_output: str, ground_truth: str) -> float:
    def get_ngrams(tokens: List[str], n: int = 4) -> Counter:
        return Counter([' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)])
    
    candidate_tokens = preprocess_text(workflow_output)
    reference_tokens = preprocess_text(ground_truth)
    
    # TF-IDF weight
    doc_freq = Counter()
    doc_freq.update(get_ngrams(reference_tokens))
    
    # n-gram
    cand_vec = get_ngrams(candidate_tokens)
    ref_vec = get_ngrams(reference_tokens)
    
    # cos
    numerator = sum((cand_vec[gram] * ref_vec[gram]) / (doc_freq[gram] + 1) 
                   for gram in set(cand_vec) & set(ref_vec))
    
    denom_cand = np.sqrt(sum((cand_vec[gram] / (doc_freq[gram] + 1))**2 
                            for gram in cand_vec))
    denom_ref = np.sqrt(sum((ref_vec[gram] / (doc_freq[gram] + 1))**2 
                           for gram in ref_vec))
    
    if denom_cand * denom_ref == 0:
        return 0.0
    
    score = numerator / (denom_cand * denom_ref)
    return score * 10

@cognify.register_evaluator
def combined_evaluator(workflow_input: str, workflow_output: str, ground_truth: str) -> float:

    bleu = bleu_evaluator(workflow_input, workflow_output, ground_truth)
    meteor = meteor_evaluator(workflow_input, workflow_output, ground_truth)
    cider = cider_evaluator(workflow_input, workflow_output, ground_truth)
    
    # align weight
    weights = {
        'bleu': 0.3,
        'meteor': 0.3,
        'cider': 0.4
    }
    
    final_score = (
        bleu * weights['bleu'] + 
        meteor * weights['meteor'] + 
        cider * weights['cider']
    )
    
    return final_score

# example
if __name__ == "__main__":
    workflow_input = "test_image label caption"
    workflow_output = "There is a cat on the windowsill observing the scenery outside"
    ground_truth = "A cat is sitting on the windowsill looking outside"
    
    bleu_score = bleu_evaluator(workflow_input, workflow_output, ground_truth)
    print(f"BLEU分数: {bleu_score:.2f}/10")

    combined_score = combined_evaluator(workflow_input, workflow_output, ground_truth)
    print(f"综合评分: {combined_score:.2f}/10")