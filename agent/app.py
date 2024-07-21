import json
import os
from typing import Any, Dict, Optional, Type

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from agent.src.llms import get_llm
from agent.src.schemas import LambdaResponse, LLMConfig, RequestConfig

output_parser = StrOutputParser()


def lambda_handler(event: Optional[Any], context: Optional[Any]) -> LambdaResponse:
    body: Dict[str, Any] = json.loads(event['body'])

    # get message and history
    message: str = body['message']
    history: list = body['history']
    state: list = body['state']

    # get inputs, inputs are dict with keys as the name of the input and values as the input
    inputs: Dict[str, Any] = body['inputs']
    config: RequestConfig = body['config']
    llm_config: LLMConfig = body['llm_config']
    system_prompt: str = body['system_prompt']
    user_prompt: str = body['user_prompt']

    # initialize model
    llm = get_llm(llm_config)

    # prompt
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("user", user_prompt)
    ]).partial(history=history)

    chain = prompt | llm | output_parser

    output: str = chain.invoke(inputs)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": output,
            }
        ),
    }
