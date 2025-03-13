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



@dataclass
class Assessment(BaseModel):
    score: int

parser = PydanticOutputParser(pydantic_object=Assessment)

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
    Load dataset from paired CSV files and return [train_data, eval_data, test_data]
    '''
    TRAIN_SPLIT, EVAL_SPLIT, TEST_SPLIT = 0.6, 0.2, 0.2
    assert TRAIN_SPLIT + EVAL_SPLIT + TEST_SPLIT == 1.0, 'Splits should sum to 1.0'

    id_caption_pairs = []
    with open(os.path.join(path_to_data, 'ids.csv'), 'r') as ids_file, \
         open(os.path.join(path_to_data, 'captions.csv'), 'r') as captions_file:
        
        for id_line, caption_line in zip(ids_file, captions_file):
            img_id = id_line.strip().split('/')[-1]  
            caption = caption_line.strip()
            id_caption_pairs.append((
                os.path.join('./img/imgs', f"{img_id}.jpg"),  
                img_id,
                caption
            ))

    if len(id_caption_pairs) > nr_samples:
        id_caption_pairs = random.sample(id_caption_pairs, nr_samples)
    random.shuffle(id_caption_pairs)

    processed_data = []
    for img_path, _, caption in id_caption_pairs:  
        processed_data.append((
            {'workflow_input': img_path},
            {'ground_truth': caption}  
        ))

    split_train = int(len(processed_data) * TRAIN_SPLIT)
    split_eval = int(len(processed_data) * (TRAIN_SPLIT + EVAL_SPLIT))
    
    return [
        processed_data[:split_train],
        processed_data[split_train:split_eval],
        processed_data[split_eval:]
    ]
# def load_textcaps_data(path_to_data="TextCaps", nr_samples=5):
#     ''' 
#     Load TextCaps dataset and return [train_data, eval_data, test_data].
    
#     Expected format:
#     - Input: {'query': 'caption', 'img_path': 'path_to_image'}
#     - Label: {'ground_truth': 'caption'}
#     '''
#     TRAIN_SPLIT, EVAL_SPLIT, TEST_SPLIT = 0.6, 0.2, 0.2 #0.8, 0.1, 0.1
#     assert TRAIN_SPLIT + EVAL_SPLIT + TEST_SPLIT == 1.0, 'Splits should sum to 1.0'
    
#     ANNOTATION_FILE_PATH = os.path.join(path_to_data, 'TextCaps_Annotations.json')
#     IMAGE_FOLDER = os.path.join(path_to_data, 'images') 

#     # Load annotations
#     with open(ANNOTATION_FILE_PATH, 'r') as f:
#         dataset = json.load(f)

#     # Extract image IDs and captions
#     all_entries = dataset["data"]
#     if len(all_entries) > subset_size:
#         all_entries = random.sample(all_entries, subset_size)  

#     # Shuffle before taking the final sample
#     random.shuffle(all_entries)  
#     selected_entries = all_entries[:nr_samples]  

#     subset_data = []
#     for entry in selected_entries:
#         input_sample = {
#             'workflow_input': os.path.join(IMAGE_FOLDER, f"{entry['image_id']}.jpg")
#         }
#         ground_truth = {
#             'ground_truth': entry["caption_str"]
#         }
#         subset_data.append((input_sample, ground_truth))


#     print(f"Total selected samples: {len(subset_data)}")

#     idx_train = int(len(subset_data) * TRAIN_SPLIT)
#     idx_eval = int(len(subset_data) * (TRAIN_SPLIT + EVAL_SPLIT))
    
#     train_data, eval_data, test_data = (
#         subset_data[:idx_train], 
#         subset_data[idx_train:idx_eval], 
#         subset_data[idx_eval:]
#     )

#     print(len(train_data), len(eval_data), len(test_data))

#     return [train_data, eval_data, test_data]

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