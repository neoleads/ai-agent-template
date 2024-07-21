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


def get_response(
        inputs, system_prompt, user_prompt, history,
        connection, api_base, api_key, model, temperature,
        split_type, min_length_output, max_length_output
):
    body = {
        "inputs": {item[0]: item[1] for item in inputs.values},
        "history": chat_to_oai_message(history),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "llm_config": {
            "connection": connection,
            "api_base": api_base,
            "api_key": api_key,
            "model": model,
            "temperature": temperature,
        },
        "config": {
            "split_type": split_type,
            "min_length_output": min_length_output,
            "max_length_output": max_length_output,
        },
    }

    response = lambda_handler({"body": json.dumps(body)}, None)
    assistant_message = json.loads(response["body"])["message"]
    return history + [(user_prompt, assistant_message)]


css = """
.chatbot-area {max-width: 100vw; max-height: 100vh;}
.logo-img img {width: 100px;}
.table {overflow: auto;}
"""

with gr.Blocks(css=css, elem_classes="chatbot-area") as demo:
    with gr.Row():
        with gr.Column(scale=12):
            gr.HTML("<center>"
                    + "<h1>AI Agent Template</h2></center>")
            gr.Markdown("<center>Experiment with AI agents</center>")
        gr.Image(value='assets/neoleads-logo.svg', elem_classes="logo-img", container=False)

    with gr.Row():
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                avatar_images=("assets/456322.webp", "assets/neoleads-logo.svg"),
                show_copy_button=True,
                height=600,
            )

            with gr.Row():
                with gr.Column(scale=4):
                    clear = gr.Button("🗑️ Clear All Message", variant='secondary')
                with gr.Column(scale=4):
                    submitBtn = gr.Button("\n💬 Send\n", size="lg", variant="primary")

        with gr.Column(scale=4):
            with gr.Tab(label="Inputs"):
                inputs = gr.Dataframe(
                    headers=["Key", "Value"],
                    datatype=["str", "str"],
                    row_count=(1, "dynamic"),  # Start with 1 row, allow dynamic addition
                    col_count=(2, "fixed"),  # Fixed number of columns
                    interactive=True,
                    label="Create Inputs",
                    wrap=True,
                    value=[["webinar_transcript", "replace with your transcript"]],
                )
                gr.Markdown("""In Prompts you can use inputs in {} format.""")
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    lines=6,
                    max_lines=10,
                    interactive=True,
                    value="""
                    You are an expert webinar summarizer who distills key learnings from webinar transcripts.

                    Your webinar summaries follow this structure:
                    
                    [1-2 sentence overview of webinar topic]
                    
                    [Bulleted list of 3-5 key learnings from the webinar]
                    
                    [1-2 closing sentences]
                    """
                )
                user_prompt = gr.Textbox(
                    label="User Prompt",
                    lines=6,
                    max_lines=10,
                    interactive=True,
                    value="""Webinar transcript: {webinar_transcript}

                    Write a concise webinar summary focusing on the key learnings from the webinar transcript 
                    provided. Follow the webinar summary structure outlined above."""
                )
            with gr.Tab(label="LLM Configuration"):
                connection = gr.Dropdown(
                    choices=["NVIDIA", "BEDROCK"],
                    label="Connection",
                    value="BEDROCK",
                    interactive=True
                )
                api_base = gr.Textbox(
                    label="API Base",
                    interactive=True
                )
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    interactive=True
                )
                model = gr.Dropdown(
                    choices=[
                        "meta/llama3-70b-instruct",
                        "anthropic.claude-3-sonnet-20240229-v1:0"
                    ],
                    label="Model",
                    value="anthropic.claude-3-sonnet-20240229-v1:0",
                    interactive=True
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.50,
                    step=0.01,
                    interactive=True,
                    label="Temperature",
                )

            with gr.Tab(label="Extra Configuration"):
                split_type = gr.Dropdown(
                    choices=["character", "word"],
                    label="Connection",
                    value="character",
                    interactive=True
                )
                min_length_output = gr.Number(
                    label="Min Length Output",
                    value=200,
                    interactive=True
                )
                max_length_output = gr.Number(
                    label="Max Length Output",
                    value=2500,
                    interactive=True
                )

    submitBtn.click(
        get_response,
        [
            inputs, system_prompt, user_prompt, chatbot,
            connection, api_base, api_key, model, temperature,
            split_type, min_length_output, max_length_output
        ], chatbot, queue=False)

    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
