import base64
import dotenv
import cognify
import logging
import os

# Set logging level
logging.basicConfig(level=logging.INFO)
dotenv.load_dotenv()

##### ----- Helper Functions ----- #####
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

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

# Create Caption agent
caption_agent = cognify.Model(
    agent_name='image_caption',
    system_prompt=CAPTION_PROMPT,
    input_variables=[
        cognify.Input(name='image', image_type='png')
    ],
    output=cognify.OutputLabel(name='caption'),
    lm_config=caption_lm_config
)

##### ----- Agent Function ----- #####
def generate_caption(image_path):
    information = {
        'image': encode_image(image_path)  # base64 encoded image
    }
    caption = caption_agent(inputs=information)
    return caption

##### ----- Workflow Definition ----- #####
@cognify.register_workflow
def vlm_workflow(workflow_input):
    caption = generate_caption(workflow_input)
    return {
        "workflow_output": caption
    }

if __name__ == "__main__":
    image_file = "test_image.jpg"  # TODO: replace with img folder
    output = vlm_workflow(image_file)
    print(output)