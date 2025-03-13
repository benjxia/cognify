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
def load_textcaps_data(path_to_data="./img", nr_samples=5):
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
    # cognify.LMConfig(model='gpt-4o-mini', kwargs={'temperature': 0, 'max_tokens': 300}),
    cognify.LMConfig(model='gpt-4o', kwargs={'temperature': 0, 'max_tokens': 300}),
]

search_settings = default.create_search(
    model_selection_cog=model_configs,
    opt_log_dir='vlm_opt_log',
)