import json
import dotenv

dotenv.load_dotenv()

import gradio as gr

from agent.app import lambda_handler


def chat_to_oai_message(chat_history):
    """
   Convert chat history to OpenAI message format.

   Args:
       chat_history (list): A list of lists, where each inner list contains the user's message and the assistant's response.

   Returns:
       list: A list of dictionaries in the OpenAI message format.
   """
    messages = []
    for msg in chat_history:
        messages.append(
            {
                "content": msg[0].split()[0] if msg[0].startswith("exitcode") else msg[0],
                "role": "user",
            }
        )
        messages.append({"content": msg[1], "role": "assistant"})
    return messages


def random_response(message, history):
    event = {
        "body": json.dumps({
            "message": message,
            "history": chat_to_oai_message(history),
        })
    }
    response = lambda_handler(event, None)
    return json.loads(response["body"])["message"]


demo = gr.ChatInterface(random_response)

if __name__ == "__main__":
    demo.launch()
