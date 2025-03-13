import os 
import random
import cognify
import json
import numpy as np
import os
os.environ['OPENAI_API_KEY'] = ...
os.environ['COGNIFY_TELEMETRY'] = "false"

DEFAULT_NR_SAMPLES = 50



############# ------- Data Loader ------- #############
@cognify.register_data_loader
def load_data(path_to_data = "VQA-sub-sampled", nr_samples=DEFAULT_NR_SAMPLES):
    ''' 
    Should return a list [train_data, eval_data, test_data]
    Each train_data, eval_data, test_data should be in form
    data = (input, label). Which in our case will be
    
    label should be the ground truth annottation
    input should be dictionaity that annottated workflow() func can handle
    
    '''
    TRAIN_SPLIT, EVAL_SPLIT, TEST_SPLIT = 0.8, 0.1, 0.1
    assert TRAIN_SPLIT + EVAL_SPLIT + TEST_SPLIT == 1.0, 'Splits should sum to 1.0'

    ANNOTTATION_FILE_PATH = os.path.join(path_to_data, 'v2_mscoco_val2014_annotations_subsampled.json')
    QUESTIONS_FILE_PATH = os.path.join(path_to_data, 'v2_OpenEnded_mscoco_val2014_questions_subsampled.json')
    IMAGE_FOLDER = os.path.join(path_to_data, 'val2014') # image in form "COCO_val2014_000000000827.jpg"

    def find_question_for_img_id(image_id: int, questions: list[dict]):
        #TODO: use a dict for faster lookup
        for question in questions:
            if question['image_id'] == image_id:
                return question
        # raise ValueError, indicating the sub-sampling was not done correctly
        raise ValueError('No question found for image_id: {}'.format(image_id))
    
    def find_annotation_forquestion_id(image_id, question_id, annotations):
        #TODO: use a dict for faster lookup
        for annotation in annottations:
            if annotation['image_id'] == image_id and annotation['question_id'] == question_id:
                return annotation
        # raise ValueError, indicating the sub-sampling was not done correctly
        raise ValueError('No annotation found for image_id: {} and question_id: {}'.format(image_id, question_id))
    
    def parse_annotation(annottation: dict):
        ''' 
        from {'image_id': 0, 'question_id': 0, 'answer_type': '...', 'answers': ['...']} to {'answer': '...'} 
        there are multiple answers, all similar to each other. But since we'll be using llm_as_judge, 
        it'll account for similartiy by itself.

        #TODO: Think about handling multiple answers
        '''
        return annottation['answers'][0]["answer"]


    all_data = [] # in form tuple(path_to_image, question, annotation)
    annottations = json.load(open(ANNOTTATION_FILE_PATH, 'r'))
    questions = json.load(open(QUESTIONS_FILE_PATH, 'r'))
    images = os.listdir(IMAGE_FOLDER)

    # pair each image with a question and an annotation
    # randomize images
    random.shuffle(images) 
    for i in range(nr_samples):
        image_path = images[i]
        image_id = int(image_path.split('_')[-1].split('.')[0])

        # question in form {'image_id': 0, 'question': '...'}
        question = find_question_for_img_id(image_id, questions) 
        # annotation in form {'image_id': 0, 'question_id': 0, 'answer_type': '...', 'answers': ['...']}
        annotation = find_annotation_forquestion_id(image_id, question['question_id'], annottations) 

        # populate tuple and append to data
        question = question['question']; annotation = parse_annotation(annotation)
        all_data.append((os.path.join(IMAGE_FOLDER, image_path), question, annotation))
    
    # transform all data into [tuple(input, label)] 
    # where input = {'query': 'query, img_path: 'path_to_image'} and label = {'ground_truth': 'annotation'}
    all_data = [( 
        # workflow input
        {'query': question, 
        'img_path': image_path}, 

        # ground truth
        {'ground_truth': annotation}) 
        for image_path, question, annotation in all_data]
    print("len(all_data): ", len(all_data))

    idx_train, idx_eval, idx_test = len(all_data)*TRAIN_SPLIT, len(all_data)*(TRAIN_SPLIT+EVAL_SPLIT), len(all_data)
    idx_train, idx_eval, idx_test = int(idx_train), int(idx_eval), int(idx_test)
    train_data, eval_data, test_data = all_data[:idx_train], all_data[idx_train:idx_eval], all_data[idx_eval:idx_test]
    return [train_data, eval_data, test_data]


############# ------- Evaluator ------- #############
import cognify

from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the model
import dotenv
dotenv.load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Force agent to respond with a score
from langchain.output_parsers import PydanticOutputParser
class Assessment(BaseModel):
    score: int
    
parser = PydanticOutputParser(pydantic_object=Assessment)

@cognify.register_evaluator
def llm_judge(workflow_output, ground_truth):
    evaluator_prompt = """
You are a dictionary evaluator. Your task is to evaluate whether two sentences describe the same thing.
Please rate the answer with a score between 0 to 2. Where 0 is completely wrong, 1 is similar.
Output a short single number. Either 0, 1 or 2. Don't write anything else. Only output a single integeer value
    """
    evaluator_template = ChatPromptTemplate.from_messages(
        [
            ("system", evaluator_prompt),
            ("human", "sentence 1:\n{sentence_1}\n\nsentence 2:\n{sentence_2}\n\nYou response format:\n{format_instructions}\n")
        ]
    )
    evaluator_agent = evaluator_template | model | parser
    assess = evaluator_agent.invoke(
        {
            "sentence_1": workflow_output,
            "sentence_2": ground_truth, 
            "format_instructions": parser.get_format_instructions()
        }
    )
    return assess.score


############# ------- Optimizer ------- #############
from cognify.hub.search import default

model_configs = [
    # OpenAI models
    cognify.LMConfig(model='gpt-4o-mini', kwargs={'temperature': 0, 'max_tokens': 300}),
    cognify.LMConfig(model='gpt-4o', kwargs={'temperature': 0, 'max_tokens': 300}),
]

search_settings = default.create_search(
    model_selection_cog=model_configs
)