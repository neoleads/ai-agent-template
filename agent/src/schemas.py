from enum import Enum

from pydantic import BaseModel, ValidationError, validator
from typing import TypedDict, Dict


class LLMConnection(str, Enum):
    nvidia = "NVIDIA"
    aws = "BEDROCK"


class SplitType(str, Enum):
    word = "word"
    char = "char"


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
    max_token: int


class RequestConfig(BaseModel):
    """
    Advanced configuration for the model
    """
    split_type: SplitType
    min_token: int
