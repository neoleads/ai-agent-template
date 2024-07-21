import json
import dotenv

dotenv.load_dotenv()

import gradio as gr


# from agent.app import lambda_handler


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
    response = {"body": json.dumps({"message": "Hello, how can I help you today?"})}
    return json.loads(response["body"])["message"]


css = """
.logo-img img {width: 100px;}
.table {overflow: auto;}
"""

with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column(scale=12):
            gr.HTML("<center>"
                    + "<h1>AI Agent Template</h2></center>")
            gr.Markdown("<center>Experiment with AI agents</center>")
        gr.Image(value='assets/neoleads-logo.svg', elem_classes="logo-img", container=False)

    with gr.Row():
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                bubble_full_width=False,
                avatar_images=("assets/456322.webp", "assets/neoleads-logo.svg"),
                render=False,
                height=600,
            )

            txt_input = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Enter text and press enter",
                container=False, render=False,
                autofocus=True,
            )
            gr.ChatInterface(
                random_response,
                chatbot=chatbot,
                textbox=txt_input,
                examples=[
                    ["tell me face value of 20 Microns."],
                    ["Create A Report on 20 Microns."]
                ],
            )
        with gr.Column(scale=4):
            with gr.Tab(label="Inputs"):
                key_value_input = gr.Dataframe(
                    headers=["Key", "Value"],
                    datatype=["str", "str"],
                    row_count=(1, "dynamic"),  # Start with 1 row, allow dynamic addition
                    col_count=(2, "fixed"),  # Fixed number of columns
                    interactive=True,
                    label="Create Inputs"
                )
                gr.Markdown("""In Prompts you can use inputs in {} format.""")
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    lines=6
                )
                user_prompt = gr.Textbox(
                    label="User Prompt",
                    lines=6
                )
            with gr.Tab(label="LLM Configuration"):
                connection = gr.Dropdown(
                    choices=["NVIDIA", "BEDROCK"],
                    label="Connection",
                    value="BEDROCK"
                )
                api_base = gr.Textbox(
                    label="API Base"
                )
                api_key = gr.Textbox(
                    label="API Key",
                    type="password"
                )
                model = gr.Dropdown(
                    choices=["meta/llama3-70b-instruct"],
                    label="Model",
                    value="meta/llama3-70b-instruct"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.30,
                    step=0.01,
                    interactive=True,
                    label="Temperature",
                )
                max_length_tokens = gr.Slider(
                    minimum=0,
                    maximum=4096,
                    value=360,
                    step=4,
                    interactive=True,
                    label="Max Generation Tokens",
                )
                split_type = gr.Dropdown(
                    choices=["Character", "words"],
                    label="Connection",
                    value="Character"
                )

            clear = gr.Button("🗑️ Clear All", variant='secondary')

if __name__ == "__main__":
    demo.launch()
