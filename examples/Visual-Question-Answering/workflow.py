import argparse
import base64
import json
import re
import uuid
import dspy

from tqdm import tqdm
# from agents.query_expansion_agent.agent import QueryExpansionAgent, query_expansion_agent
# from agents.plot_agent.agent import PlotAgent, PlotAgentModule
# from agents.visual_refine_agent import VisualRefineAgent
# from agents.utils import is_run_code_success, run_code, get_code
# from agents.dspy_common import OpenAIModel
# from agents.config.openai import openai_kwargs
import logging
import os
import shutil
import glob
import sys
import cognify
import dotenv

# set to info level logging
logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()


##### ----- Define Agent (LLM call) that'll be utilized in workflow ----- #####
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

VQA_PROMPT = """
You are an expert in answering questions about related image. Given a user query, and provided question answer the question about the image. 
Your task is to provide a short detailed answer to the question based on the information in the image. 
Analyze the image carefully to identify relevant details that can help you answer the question. 
Consider the context, objects, and relationships within the image to provide a comprehensive response. 
Your answer should be clear, concise, and directly address the question asked.
Your answer should short and concise, don't use more than 5 words. Aim to use 2 or 3 words if possible.
"""

visual_refine_lm_config = cognify.LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs= {'temperature': 0.0,}
)

vqa_agent = cognify.Model(agent_name='vwa_refinement', 
                          system_prompt=VQA_PROMPT,
                          input_variables=[cognify.Input(name='query'), 
                                           cognify.Input(name='image', image_type='png')],
                          output=cognify.OutputLabel(name='annottation'),
                          lm_config=visual_refine_lm_config)

def run_vqa_agent(image_path, query):
    information = {
        'query': query,
        'image': encode_image(image_path) # base 64 encoded image
    }
    visual_feedback = vqa_agent(inputs=information)
    return visual_feedback


##### ----- Define Workflow ----- #####
@cognify.register_workflow
def mainworkflow(query, img_path):
    return {
        "workflow_output": run_vqa_agent(img_path, query)
    }


    
