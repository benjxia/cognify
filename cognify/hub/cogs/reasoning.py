from abc import ABCMeta
from typing import List, Union
from cognify.hub.cogs.common import CogBase, CogLayerLevel, OptionBase, NoChange
from cognify.llm import Model, StructuredModel, litellm_completion
from cognify.llm.model import APICompatibleMessage
from litellm import ModelResponse
import copy

import logging

logger = logging.getLogger(__name__)


class LMReasoning(CogBase):
    level = CogLayerLevel.NODE

    def __init__(
        self,
        options: list[OptionBase],
        name: str = "reasoning",
        default_option: Union[int, str] = 0,
        module_name: str = None,
        inherit: bool = True,
    ):
        return super().__init__(name, options, default_option, module_name, inherit)

    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, default_option, options = (
            data["name"],
            data["module_name"],
            data["default_option"],
            data["options"],
        )
        options = [
            ReasonThenFormat.registry[dat["type"]].from_dict(dat)
            for name, dat in options.items()
        ]
        return cls(
            name=name,
            options=options,
            default_option=default_option,
            module_name=module_name,
        )


class ReasoningOptionMeta(ABCMeta):
    registry: dict[str, type] = {"NoChange": NoChange}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        cls.registry[name] = new_cls
        return new_cls


class ReasonThenFormat(OptionBase, metaclass=ReasoningOptionMeta):
    @classmethod
    def direct_apply(cls, lm_module: Model):
        reasoning = cls()
        reasoning.apply(lm_module)
        return reasoning

    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        """Produce reasoning steps for the given chat prompt messages"""
        raise NotImplementedError

    def aggregate_reasoning_steps(self, responses: List[ModelResponse]) -> str:
        agg_messages = []
        for response in responses:
            agg_messages.append(f"\n: {response.choices[0].message.content}")
        return "\n".join(agg_messages)

    def forward(
        self, lm_module: Model, messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        """
        If the orignal output has certain format, applying additional reasoning steps will break down
        it into two phases, first one allows free generation along with reasoning steps, and the second
        one will the formatting step
        """

        model: str = model_kwargs.pop("model")
        responses = []

        messages.append(
            {
                "role": "user",
                "content": "Don't give your final response to the instruction directly. We can start with some reasoning first.\n",
            }
        )
        reasoning_step_responses: List[ModelResponse] = self.reasoning_step(
            model, copy.deepcopy(messages), model_kwargs
        )

        responses.extend(reasoning_step_responses)
        rationale = self.aggregate_reasoning_steps(reasoning_step_responses)
        lm_module.rationale = rationale

        messages.append({"role": "assistant", "content": rationale})
        if lm_module.contains_custom_format_instructions():
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on the reasoning, now please only give {lm_module.get_output_label_name()} as your final response, according to the following instructions:\n{lm_module.get_custom_format_instructions_if_any()}",
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": f"Based on the reasoning, now please form {lm_module.get_output_label_name()} as your final response.",
                }
            )

        full_messages = [lm_module.system_message.to_api()] + messages
        if isinstance(lm_module, StructuredModel):
            response = litellm_completion(
                model,
                full_messages,
                model_kwargs,
                response_format=lm_module.output_format.schema,
            )
            responses.append(response)
        else:
            response = litellm_completion(model, full_messages, model_kwargs)
            responses.append(response)
        return responses

    def apply(self, lm_module: Model):
        lm_module.reasoning = self
        return lm_module

    @classmethod
    def from_dict(cls, data: dict):
        return cls()


class ZeroShotCoT(ReasonThenFormat):
    def __init__(self):
        super().__init__("ZeroShotCoT")

    def _get_cost_indicator(self):
        return 2.0

    def describe(self):
        desc = """
        - ZeroShotCoT -
        Return step-by-step reasoning for the given chat prompt messages.

        Reasoning Prompt:
            Let's solve this problem step by step before giving the final response.
        """
        return desc

    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        chat_messages.append(
            {
                "role": "user",
                "content": "Let's solve this problem step by step before giving the final response\n",
            }
        )
        response = litellm_completion(model, chat_messages, model_kwargs)
        return [response]


class PlanBefore(ReasonThenFormat):
    def __init__(self):
        super().__init__("PlanBefore")

    def _get_cost_indicator(self):
        return 3.0

    def describe(self):
        desc = """
        - PlanBefore -
        Similar to the planner in the LLMCompiler paper. Plan sub-tasks and synthesize a response for each sub-task as the rationale. Focus more on the runtime query complexity.

        Reasoning Prompt:
            Let's first break down the task into several simpler sub-tasks that each covers different aspect of the original task. Clearly state each sub-question and provide your response to each one of them.
        """
        return desc

    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        # TODO: make this a workflow and parallelize the reasoning steps
        chat_messages.append(
            {
                "role": "user",
                "content": "Let's first break down the task into several simpler sub-tasks that each covers different aspect of the original task. Clearly state each sub-question and provide your response to each one of them.",
            }
        )
        response = litellm_completion(model, chat_messages, model_kwargs)
        return [response]

class VisionPlanning(ReasonThenFormat):
    def __init__(self):
        super().__init__("VisionPlanning")

    def _get_cost_indicator(self):
        return 3.0

    def describe(self):
        desc = """
        - Vision Planning -
        Return step-by-step reasoning for the given chat prompt messages.

        Reasoning Prompt:
            Let's solve this problem step by step before giving the final response.
        """
        return desc

    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        has_image = False
        for message in chat_messages:
            for item in message["content"]:
                if item["type"] == "image_url":
                    has_image = True
                    break
            if has_image:
                break

        if not has_image:
            # TODO: raise warning message about missing image
            # TODO: or default to some other reasoning cog option
            return []

        responses = []

        reasoning_messages = chat_messages.copy()

        reasoning_messages.append(
            {
                "role": "user",
                "content": "Generate a detailed caption about the image(s) provided.\n",
            }
        )

        caption_response = litellm_completion(model, reasoning_messages, model_kwargs)
        reasoning_messages.pop(-1) # Remove prompt from reasoning_messages
        responses.append(caption_response)

        # Add model response to context
        reasoning_messages.append(
            {
                "role": "assistant",
                "content": "Image Caption: " + caption_response.choices[0].message.content + "\n",
            }
        )

        reasoning_messages.append(
            {
                "role": "user",
                "content": "Generate sub-questions about the image(s) provided. You may use the provided image caption. Clearly state each sub-question and provide your response to each one of them.\n",
            }
        )

        responses.append(litellm_completion(model, reasoning_messages, model_kwargs))

        return responses


class VLMQueryRewriting(ReasonThenFormat):
    def __init__(self):
        super().__init__("VLMQueryRewriting")

    def _get_cost_indicator(self):
        return 3.0

    def describe(self):
        desc = """
        - VLM Query Rewriting -
        Get a Caption of an image, and re-write the query based on the caption. 
        Based on the following paper: https://arxiv.org/pdf/2311.09050

        No promp engineering per se
        """
        return desc

    def reasoning_step(
        self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict
    ) -> List[ModelResponse]:
        has_image = False
        for message in chat_messages:
            for item in message["content"]:
                if item["type"] == "image_url":
                    has_image = True
                    break
            if has_image:
                break

        if not has_image:
            # TODO: raise warning message about missing image
            # TODO: or default to some other reasoning cog option
            return []

        responses = []

        reasoning_messages = chat_messages.copy()

        reasoning_messages.append(
            {
                "role": "user",
                "content": "Generate a detailed caption about the image(s) provided.\n",
            }
        )

        caption_response = litellm_completion(model, reasoning_messages, model_kwargs)
        
        caption = caption_response.choices[0].message.content
        print("caption", caption)

        initial_query = model_kwargs["prompt"]
        print("initial_query", initial_query)

        new_query = reWriteQuery(caption, initial_query)
        print("new_query", new_query)

        # Create a new chat message with the rewritten query
        rewritten_query_message = {
            "role": "user",
            "content": new_query
        }

        # Get a completion from the model using the rewritten query
        model_kwargs.pop("prompt") # Remove prompt from model_kwargs
        response = litellm_completion(model, [rewritten_query_message], model_kwargs)
        
        return [response]



# TODO: Move the following to a seperate file and import here
# TODO: Rather than calling a certain type of GPT via Langchain directly, call the model specified within the Cog instead 
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, validator
import dotenv
import os
from typing import List


from constituent_treelib import ConstituentTree, Language
# Define the language for the sentence as well as for the spaCy and benepar models
language = Language.English
# Define which specific SpaCy model should be used (default is Medium)
spacy_model_size = ConstituentTree.SpacyModelSize.Medium
# Create the pipeline (note, the required models will be downloaded and installed automatically)
nlp = ConstituentTree.create_pipeline(language, spacy_model_size)
# Your sentence


def reWriteQuery(caption: str, question: str) -> str:
    ''' 
    The "main meat" function of VLM Query Rewriting cog

    Given
    * a caption like "a clock above a building with clouds and lightning background"
    * a question like "what unpleasant emotion does this weather phenomenon evoke?"

    Return
    * the best question rewrite like 'what unpleasant emotion does lightning background evoke?'
    This is useful since it substitutes a vague question with a more specific one based on the image caption
    '''
    caption_constituents = get_constituents(caption)
    question_constituents = get_constituents(question)
    questions = get_all_possible_question_rewrites(question, caption_constituents, question_constituents)
    best_question = choose_the_best_questions(question, questions)
    return best_question

def get_constituents(sentence: str) -> list[str]:
    '''Given a sentence like "a clock above a building with clouds and lightning background" 
    return all possible constituents like ["a clock", "a building", "clouds", "lightning background"]'''
    tree = ConstituentTree(sentence, nlp)
    all_phrases = tree.extract_all_phrases(min_words_in_phrases=1)
    return all_phrases["NP"]

def get_all_possible_question_rewrites(question : str, 
                                       caption_constituents : list[str], 
                                       question_constituents : list[str]) -> list[str]:
    
    '''
    Given 
    * a question like "what unpleasant emotion does this weather phenomenon evoke?" 
    * caption constituents like ["a clock", "a building", "clouds", "lightning background"]
    * question constituents like ["the color of the clock", "the time on the clock", "the material of the clock"], 
    
    return all possible questions like
    [
        'what unpleasant emotion does clouds evoke?', 
        'what unpleasant emotion does a clock evoke?', 
        'what unpleasant emotion does lightning background evoke?'
    ]
    '''
    questions = []
    for q in question_constituents:
        for c in caption_constituents:
            new_sentence = question.replace(q, c)
            questions.append(new_sentence)

    return questions

def choose_the_best_questions(original_question: str, all_possible_sentences: List[str]) -> str:
    ''' 
    Given 
    * the original question like "what unpleasant emotion does this weather phenomenon evoke?"
    * a list of all possible questions rewrites like:
        [
            'what unpleasant emotion does clouds evoke?', 
            'what unpleasant emotion does a clock evoke?', 
            'what unpleasant emotion does lightning background evoke?'
        ]

    Return 
    * the best question rewrite like 'what unpleasant emotion does lightning background evoke?'

    The definition of "best" is the one that is most relevant to the original question 
    (eg. the subtitution of constituents makes sense in the context of the question and preserves the original intent)

    '''

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # sentence_assessments is a list of scores (scores are wrapped in RelevanceAssessment objects)
    sentence_assessments = assess_relevance_batch(original_question, all_possible_sentences, model)

    # Find the sentence with the highest relevance score
    best_sentence = None
    best_score = -1
    best_index = -1

    for i, assessment in enumerate(sentence_assessments):
        if assessment.score > best_score:
            best_score = assessment.score
            best_sentence = all_possible_sentences[i]
            best_index = i

    print(f"LLM chose index {best_index} with score {best_score} because: {sentence_assessments[best_index].reason}")

    if best_sentence is None:
        print("No sentences were assessed. Returning the first question.")
        return all_possible_sentences[0]

    return best_sentence




# Below are just a hardcoded call to chatGPT using Langchain 
# So we can ensure it returns a correct pedantric numerical output

class RelevanceAssessment(BaseModel):
    score: int
    reason: str

    @validator('score')
    def score_must_be_in_range(cls, value):
        if not 1 <= value <= 10:
            raise ValueError('Score must be between 1 and 10')
        return value

class BatchRelevanceAssessment(BaseModel):
    assessments: List[RelevanceAssessment]

parser = PydanticOutputParser(pydantic_object=BatchRelevanceAssessment)


def assess_relevance_batch(original_question: str, new_questions: List[str], model: ChatOpenAI) -> List[RelevanceAssessment]:
    """Assesses the relevance of a batch of new questions compared to the original."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an expert in language and reasoning. 
             Your task is to assess how well each new question retains the meaning and intent of the original question,
             considering a substitution that has been made.
             For each new question, compare the original question to the new question. Think step by step about whether the substitution makes sense
             in the context of the question and whether it preserves the original intent.
             Provide a score from 1 to 10 for each new question, where 1 means the new question is completely irrelevant to the original,
             and 10 means the new question is perfectly relevant and retains the original intent.
             Explain your reasoning for each question in a short sentence.

             Return a JSON array of objects, where each object has a "score" and a "reason" field.
             The scores must be between 1 and 10 (inclusive).
             Your response format should be: {format_instructions}
             """),
            ("human", "Original question: {original_question}\nNew questions:\n{new_questions}")
        ]
    )

    chain = prompt | model | parser

    # Format the new questions for the prompt
    formatted_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(new_questions)])

    input = {"original_question": original_question,
             "new_questions": formatted_questions,
             "format_instructions": parser.get_format_instructions()}

    output = chain.invoke(input)
    return output.assessments