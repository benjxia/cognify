from abc import ABCMeta
import copy
from typing import Literal, Optional, Union
import uuid
import dataclasses
import heapq
import os
import logging
from itertools import combinations

from cognify.llm.litellm_wrapper import litellm_completion
from cognify.llm.model import APICompatibleMessage


logger = logging.getLogger(__name__)

from cognify.llm import Model, Demonstration
from cognify.llm.prompt import FilledInput, Input
from cognify.hub.cogs.common import (
    CogBase,
    EvolveType,
    CogLayerLevel,
    OptionBase,
    NoChange,
)

from cognify.llm.prompt import (
    Input,
    FilledInput,
    CompletionMessage,
    Demonstration,
    Content,
    TextContent,
    ImageContent,
    get_image_content_from_upload,
)

from cognify.hub.cogs.utils import dump_params, load_params
from typing import List


class GenerateContext(CogBase):
    level = CogLayerLevel.NODE

    def __init__(
        self,
        options: list[OptionBase],
        name: str = "GenerateContext",
        default_option: Union[int, str] = 0,
        module_name: str = None,
        inherit: bool = True,
    ):
        super().__init__(
            name,
            options,
            0,
            module_name,
            inherit=inherit
        )

    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, default_option, options = (
            data["name"],
            data["module_name"],
            data["default_option"],
            data["options"],
        )

        options = [
            GenContextMeta.registry[dat["type"]].from_dict(dat)
            for name, dat in options.items()
        ]

        return cls(
            name=name,
            options=options,
            default_option=default_option,
            module_name=module_name,
        )


class GenContextMeta(ABCMeta):
    registry: dict[str, type] = {"NoChange": NoChange}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        cls.registry[name] = new_cls
        return new_cls

class GenerateImageContext(OptionBase, metaclass=GenContextMeta):
    def generate_caption(self, model: str, images: List[ImageContent], model_kwargs: dict) -> str:
        CAPTION_PROMPT = "Generate a detailed caption about the image(s) provided. If there are multiple images, give a description in a numbered list.\n"
        msg_contents = [
            {
                "type": "text",
                "text": CAPTION_PROMPT
            }
        ] + images

        messages = [
            {
                "role": "user",
                "content": msg_contents
            }
        ]

        caption_response = litellm_completion(model, messages, model_kwargs)
        return caption_response.choices[0].message.content

    def generate_context(self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict) -> List[APICompatibleMessage]:
        raise NotImplementedError

    def get_images(self, chat_messages: List[APICompatibleMessage]) -> List[ImageContent]:
        images = []
        for message in chat_messages:
            # Skip text-only messages
            if isinstance(message["content"], str):
                continue
            for item in message["content"]:
                if item["type"] == "image_url":
                    images.append(item)
        return images

    def forward(self, lm_module: Model, chat_messages: List[APICompatibleMessage], model_kwargs: dict) -> List[APICompatibleMessage]:
        kwargs = copy.deepcopy(model_kwargs)
        model: str = kwargs.pop("model")
        return self.generate_context(model, chat_messages, kwargs)

    def apply(self, lm_module: Model):
        lm_module.gencontext = self
        return lm_module



class DefaultImageContext(GenerateImageContext, metaclass=GenContextMeta):
    def __init__(self):
        super().__init__("DefaultImageContext")

    def describe(self):
        desc = "- Generate Additional Image Context - \n"
        return desc

    def _get_cost_indicator(self):
        # Rely on LLM response for precise image tokenization info
        return 2.0

    def generate_context(self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict) -> List[APICompatibleMessage]:
        images = self.get_images(chat_messages)
        caption = self.generate_caption(model, images, model_kwargs)
        chat_messages.append(
            {
                "role": "assistant",
                "content": "Image captions:\n" + caption
            }
        )
        return chat_messages

    @classmethod
    def from_dict(cls, data: dict):
        return cls()

class VisionPlanningContext(GenerateImageContext, metaclass=GenContextMeta):
    def __init__(self, prompt: Optional[str] = None):
        super().__init__("VisionPlanningContext")
        self.prompt = prompt

    def describe(self):
        desc = "- Generate Additional Image Context - \n"
        return desc

    def _get_cost_indicator(self):
        # Rely on LLM response for precise image tokenization info
        return 3.0

    def generate_context(self, model: str, chat_messages: List[APICompatibleMessage], model_kwargs: dict) -> List[APICompatibleMessage]:
        images = self.get_images(chat_messages)
        caption = self.generate_caption(model, images, model_kwargs)
        chat_caption_content = "Image captions:\n" + caption
        chat_messages.append(
            {
                "role": "assistant",
                "content": chat_caption_content
            }
        )

        SUBQUESTION_PROMPT = "Generate sub-questions about the image(s) provided. You may use the provided image caption. Clearly state each sub-question and provide your response to each one of them.\n"
        reasoning_messages = []
        reasoning_messages.append(
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": chat_caption_content
                }] + images
            }
        )
        reasoning_messages.append(
            {
                "role": "user",
                "content": SUBQUESTION_PROMPT if not self.prompt else self.prompt
            }
        )

        reasoning_response = litellm_completion(model, reasoning_messages, model_kwargs)
        chat_messages.append(
            {
                "role": "assistant",
                "content": reasoning_response.choices[0].message.content
            }
        )

        return chat_messages

    @classmethod
    def from_dict(cls, data: dict):
        return cls()

