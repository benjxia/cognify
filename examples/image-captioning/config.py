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
def load_data():
    with open("vlm_data._json", "r") as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)
    # Format to (input, output) pairs
    new_data = []
    for d in data:
        input_sample = {
            'workflow_input': d["image_path"],
        }
        ground_truth = {
            'ground_truth': d["expected_caption"],
        }
        new_data.append((input_sample, ground_truth))

    # Train, val, test split
    return new_data[0:], None, new_data[0:]

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