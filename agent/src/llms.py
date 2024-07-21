from typing import Any

from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI

from src.schemas import LLMConfig, LLMConnection


def get_llm(config: LLMConfig) -> Any:
    if config.connection == LLMConnection.nvidia:
        return ChatOpenAI(
            openai_api_base=config.api_base,
            openai_api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
        )
    elif config.connection == LLMConnection.aws:
        return ChatBedrockConverse(
            model=config.model,
            temperature=config.temperature,
        )
