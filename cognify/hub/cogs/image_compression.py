from abc import ABCMeta
from typing import Literal
import uuid
import dataclasses
import heapq
import os
import logging
from itertools import combinations


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
from cognify.hub.cogs.utils import dump_params, load_params
from typing import List


class VLMImageQuality(CogBase):
    level = CogLayerLevel.NODE

    def __init__(
        self,
        options: list[OptionBase],
        name: str = "image_downsample",
        module_name: str = None,
        inherit: bool = True,
    ):
        super().__init__(
            name,
            options,
            0,
            module_name,
            inherit=inherit,
            inherit_options=False,
        )



    def _evolve(self, eval_result) -> EvolveType:
        pass

    @classmethod
    def from_dict(cls, data: dict):
        name, module_name, default_option, options = (
            data["name"],
            data["module_name"],
            data["default_option"],
            data["options"],
        )
        options = [
            # TODO: implement this
            # ReasonThenFormat.registry[dat["type"]].from_dict(dat)
            # for name, dat in options.items()
        ]
        return cls(
            name=name,
            options=options,
            default_option=default_option,
            module_name=module_name,
        )


class ImageQualityMeta(ABCMeta):
    registry: dict[str, type] = {"NoChange": NoChange}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        cls.registry[name] = new_cls
        return new_cls

    def setting():
        pass

class LowQuality(OptionBase, metaclass=ImageQualityMeta):

    def describe(self):
        desc = "- Image Quality: Low - \n"
        return desc

    def _get_cost_indicator(self):
        # Rely on LLM response for image tokenization info
        return 1.0

    def apply(self, lm_module: Model):
        lm_module.image_downsample = "low"
        return lm_module

class HighQuality(OptionBase, metaclass=ImageQualityMeta):
    def describe(self):
        desc = "- Image Quality: High - \n"
        return desc

    def _get_cost_indicator(self):
        # Rely on LLM response for image tokenization info
        return 1.0

    def apply(self, lm_module: Model):
        lm_module.image_downsample = "high"
        return lm_module
