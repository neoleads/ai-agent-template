import json
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.llms import get_llm
from src.schemas import LambdaResponse, LLMConfig, RequestConfig

output_parser = StrOutputParser()


def lambda_handler(event: Optional[Any], context: Optional[Any]) -> LambdaResponse:
    try:
        body: Dict[str, Any] = json.loads(event['body'])
    except TypeError:
        body: Dict[str, Any] = event['body']

    # get message and history
    history: list = body['history']
    state: int = body.get('state', 0)

    # get inputs, inputs are dict with keys as the name of the input and values as the input
    inputs: Dict[str, Any] = body['inputs']
    config: RequestConfig = RequestConfig(**body['config'])
    llm_config: LLMConfig = LLMConfig(**body['llm_config'])
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
