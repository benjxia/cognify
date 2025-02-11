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

@cognify.register_data_loader
def load_textcaps_data(path_to_data="TextCaps/TextCaps_Annotations.json", limit=1000):
    """Loads a random subset of TextCaps dataset (default: 1000 images) and prepares it in Cognify format."""
    
    with open(path_to_data, "r") as file:
        dataset = json.load(file)  # Load the JSON file

    subset_data = random.sample(dataset["data"], limit)

    # Construct a mapping of image_id -> flickr_original_url
    images = {entry["image_id"]: entry["flickr_original_url"] for entry in subset_data}
    
    # Construct a mapping of image_id -> caption
    annotations = {entry["image_id"]: entry["caption_str"] for entry in subset_data}

    # Prepare data in Cognify format (input, label)
    all_data = [
        ({"query": "", "img_url": images[image_id]}, {"ground_truth": annotations[image_id]})
        for image_id in images if image_id in annotations
    ]

    # Train/eval/test split
    TRAIN_SPLIT, EVAL_SPLIT, TEST_SPLIT = 0.8, 0.1, 0.1
    random.shuffle(all_data)  

    idx_train = int(len(all_data) * TRAIN_SPLIT)
    idx_eval = int(len(all_data) * (TRAIN_SPLIT + EVAL_SPLIT))
    
    train_data, eval_data, test_data = all_data[:idx_train], all_data[idx_train:idx_eval], all_data[idx_eval:]
    
    return [train_data, eval_data, test_data]

# def load_textcaps_data(path_to_data="TextCaps", nr_samples=50):
#     ''' 
#     Load TextCaps dataset and return [train_data, eval_data, test_data].
    
#     Expected format:
#     - Input: {'query': 'caption', 'img_path': 'path_to_image'}
#     - Label: {'ground_truth': 'caption'}
#     '''
#     TRAIN_SPLIT, EVAL_SPLIT, TEST_SPLIT = 0.8, 0.1, 0.1
#     assert TRAIN_SPLIT + EVAL_SPLIT + TEST_SPLIT == 1.0, 'Splits should sum to 1.0'
    
#     ANNOTATION_FILE_PATH = os.path.join(path_to_data, 'TextCaps_Annotations.json')
#     IMAGE_FOLDER = os.path.join(path_to_data, 'images')  # Adjust based on dataset structure

#     # Load annotations
#     with open(ANNOTATION_FILE_PATH, 'r') as f:
#         dataset = json.load(f)
    
#     images = {img["image_id"]: f"{img['image_id']}.jpg" for img in dataset["data"]}
#     annotations = {entry["image_id"]: entry["caption_str"] for entry in dataset["data"]}

#     all_data = []
#     random.shuffle(annotations)  # Shuffle for randomness

#     for i in range(min(nr_samples, len(annotations))):
#         ann = annotations[i]
#         image_id = ann["image_id"]
#         image_path = os.path.join(IMAGE_FOLDER, images[image_id])
#         caption = ann["caption"]

#         # Store in required format
#         all_data.append((
#             {'query': caption, 'img_path': image_path},  # Input
#             {'ground_truth': caption}  # Label
#         ))

#     print("Total samples:", len(all_data))

#     # Split dataset
#     idx_train = int(len(all_data) * TRAIN_SPLIT)
#     idx_eval = int(len(all_data) * (TRAIN_SPLIT + EVAL_SPLIT))
    
#     train_data, eval_data, test_data = (
#         all_data[:idx_train], 
#         all_data[idx_train:idx_eval], 
#         all_data[idx_eval:]
#     )

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