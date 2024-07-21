from enum import Enum

from pydantic import BaseModel, ValidationError, validator
from typing import TypedDict, Dict


class LLMConnection(str, Enum):
    nvidia = "NVIDIA"
    aws = "BEDROCK"


class SplitType(str, Enum):
    word = "work"
    char = "character"


class LambdaResponse(TypedDict):
    statusCode: int
    body: str


class LLMConfig(BaseModel):
    """
    Advanced configuration for the model
    """
    connection: LLMConnection
    api_base: str
    api_key: str
    model: str
    temperature: float


class RequestConfig(BaseModel):
    """
    Advanced configuration for the model
    """
    split_type: SplitType
    min_length_output: int
    max_length_output: int
