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

##### ----- Agent Function ----- #####
def generate_caption(image_path):
    information = {
        'image': encode_image(image_path)  # base64 encoded image
    }
    # print("---")
    # print(image_path)
    # print(encode_image(image_path))
    # print("---")
    classification = classification_agent(inputs=information)
    return classification


##### ----- Workflow Definition ----- #####
@cognify.register_workflow
def vlm_workflow(workflow_input):
    classification = generate_caption(workflow_input)
    return {
        "workflow_output": classification
    }

import json

def transform_json(input_file="TextCaps/TextCaps_Annotations.json", output_file="output.json"):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    transformed_data = []
    for item in data['data']:
        transformed_item = {
            'image_path': item['flickr_300k_url'],
            'expected_caption': item['caption_str']
        }
        transformed_data.append(transformed_item)
    
    with open(output_file, 'w') as f:
        json.dump(transformed_data, f, indent=4)

if __name__ == '__main__':
    image_file = "./img/imgs/00001bc2c4027449.jpg" 
    if not os.path.exists(image_file):
        print(f"Error: The image file '{image_file}' does not exist.")
    else:
        output = vlm_workflow(image_file)
        print(output)