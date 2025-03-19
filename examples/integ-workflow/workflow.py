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
                                           cognify.Input(name='image', image_type='png'),
                                           cognify.Input(name='class'),
                                           cognify.Input(name='caption')],
                          output=cognify.OutputLabel(name='annottation'),
                          lm_config=visual_refine_lm_config)

##### ----- Define Agent Configuration ----- #####
CAPTION_PROMPT = """
You are an image understanding expert. Your task is to analyze an image and describe its contents clearly.
Identify key objects, actions, and any relevant context. Provide a detailed and accurate caption for the image.
"""

# Configure LLM settings
caption_lm_config = cognify.LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs={'temperature': 0.0}
)

caption_agent = cognify.Model(
    agent_name='image_caption',
    system_prompt=CAPTION_PROMPT,
    input_variables=[
        cognify.Input(name='image', image_type='png')
    ],
    output=cognify.OutputLabel(name='caption'),
    lm_config=caption_lm_config
)

##### ----- Define Agent Configuration ----- #####
CLASS_PROMPT = """
You are an image classification expert. Identify the main subject or theme of the image and pick the labels from these: "Animal,Backpack,Baked goods,Balloon,Bicycle,Bicycle wheel,Bird,Boat,Book,Boot,Bowl,Box,Boy,Bronze sculpture,Building,Cake,Car,Carnivore,Cat,Chair,Clothing,Coat,Coconut,Container,Cowboy hat,Curtain,Dairy Product,Dessert,Door,Dress,Drink,Drum,Fedora,Flag,Flower,Food,Footwear,Furniture,Girl,Glasses,Glove,Goggles,Guitar,Handbag,Hat,Headphones,Helmet,House,Houseplant,Human arm,Human beard,Human body,Human ear,Human eye,Human face,Human foot,Human hair,Human hand,Human head,Human leg,Human mouth,Human nose,Jacket,Jeans,Kitchen utensil,Kitchenware,Lamp,Land vehicle,Lantern,Light bulb,Luggage and bags,Mammal,Man,Microphone,Miniskirt,Mobile phone,Musical instrument,Musical keyboard,Necklace,Palm tree,Pastry,Person,Plant,Platter,Saucer,Scarf,Sculpture,Seafood,Shirt,Shorts,Shotgun,Skirt,Skyscraper,Sock,Sports uniform,Stairs,Street light,Suit,Sun hat,Sunglasses,Table,Tableware,Taxi,Tie,Tire,Tower,Toy,Tree,Trousers,Van,Vegetable,Vehicle,Vehicle registration plate,Watch,Watercraft,Weapon,Wheel,Window,Woman".
"""

# Configure LLM settings
caption_lm_config = cognify.LMConfig(
    custom_llm_provider='openai',
    model='gpt-4o-mini',
    kwargs={'temperature': 0.0}
)

# Create Caption agent
classification_agent = cognify.Model(
    agent_name='image_classification',
    system_prompt=CLASS_PROMPT,
    input_variables=[
        cognify.Input(name='image', image_type='png')
    ],
    output=cognify.OutputLabel(name='classification'),
    lm_config=caption_lm_config
)


def run_vqa_agent(image_path, query):
    # classification
    information = {
        'image': encode_image(image_path)  # base64 encoded image
    }
    classification = classification_agent(inputs=information)
    # caption
    information = {
        'image': encode_image(image_path)  # base64 encoded image
    }
    caption = caption_agent(inputs=information)

    # vqa
    # TODO naive integration, without cog modification
    information = {
        'query': query,
        'image': encode_image(image_path), # base 64 encoded image
        'class': classification,
        'caption': caption
    }
    visual_feedback = vqa_agent(inputs=information)
    return visual_feedback


##### ----- Define Workflow ----- #####
@cognify.register_workflow
def mainworkflow(query, img_path):
    return {
        "workflow_output": run_vqa_agent(img_path, query)
    }


    