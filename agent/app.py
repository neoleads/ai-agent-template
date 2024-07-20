#! /usr/bin/env python

import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()


def lambda_handler(event, context):
    body = json.loads(event['body'])

    # get message and history
    message = body['message']
    history = body['history']

    # initialize model
    llm = ChatOpenAI(
        openai_api_base=os.getenv('OPENAI_BASE_URL'),
        model_name=os.getenv('OPENAI_MODEL'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
    )

    # prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world class technical documentation writer."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{input}")
    ]).partial(history=history)

    chain = prompt | llm | output_parser

    output = chain.invoke({
        "input": message
    })

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": output,
            }
        ),
    }
