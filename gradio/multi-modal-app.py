import gradio as gr
import base64
import os
from openai import OpenAI
import httpx

import json


#configure default environment variables

os.environ['CDP_TOKEN'] = json.load(open("/tmp/jwt"))["access_token"]
#os.environ['OPENAI_API_BASE'] = "<ENDPOINT_URL>"
#os.environ['OPENAI_MODEL_NAME'] = "<MODEL_NAME>"


if "CDP_TOKEN" not in os.environ:
    raise ValueError("CDP_TOKEN environment variable must be set")

if "OPENAI_API_BASE" not in os.environ:
    raise ValueError("OPENAI_API_BASE environment variable must be set")

if "OPENAI_MODEL_NAME" not in os.environ:
    raise ValueError("OPENAI_MODEL_NAME environment variable must be set")

if "CUSTOM_CA_STORE" not in os.environ:
    http_client = httpx.Client()
else:
    http_client = httpx.Client(verify=os.environ["CUSTOM_CA_STORE"])

print("Starting application...")
client = OpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["CDP_TOKEN"],
    http_client=http_client,
)

ext_history = []

# Image to Base 64 Converter
def image_to_base64(image_path):
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode('utf-8')

#Generate OpenAI compatible history for multi-modal inputs
def refactor_history_for_openai(history):
    openai_history_format = []
    for messages in history:
        if isinstance(messages["content"], tuple):
            openai_history_format.append({"role": "user", "content": "uploaded an image named {0}".format(messages["content"][0])})
        else:
            openai_history_format.append(messages)
    return openai_history_format

#Run inference on model specified using OpenAI Protocol
def predict(message, history):
    oai_hist = refactor_history_for_openai(history)
    q = []
    for image in message["files"]:
        if image.startswith("http"):
          query = {"role": "user", "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image
                    }
                }
            ]}
        else:
            base64 = image_to_base64(image)
            query = {"role": "user", "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,{0}".format(base64)
                        }
                    }
                ]}
        q.append(query)
    if not history:
        q.append({"role": "user", "content": "you are a data scientist helper.  Your goal is to take images of graphs and answer the questions and comments that the user has.  If you are uncertain of your answer, respond with I need more context"})
    q.append({"role": "user", "content": message["text"]})
    response = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL_NAME"],
        messages=oai_hist + q,
        stream=True,
    )
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message



with gr.Blocks() as demo:
    gr.Markdown("<center><h1>Multi-modal chatbot</h1></center>")
    gr.Markdown("<center><h3>Powered by Llama 3.2 in AI Inference</h3></center>")
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Enter your message here...")
    upload = gr.File(file_count="multiple")
    
    def user(user_message, files, history):
        if history is None:
            history = []
        message = {"text": user_message, "files": [f.name for f in files] if files else []}
        # Convert to the format Chatbot expects (list of tuples)
        history.append((user_message, None))
        return "", None, history
    
    def bot(history):
        # Convert history to the format your predict function expects
        formatted_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
            for i, (msg, _) in enumerate(history[:-1])
        ]
        # Get the last user message
        last_message = history[-1][0]
        bot_message = None
        for response in predict({"text": last_message, "files": []}, formatted_history):
            bot_message = response
            history[-1] = (history[-1][0], bot_message)
            yield history

    msg.submit(user, [msg, upload, chatbot], [msg, upload, chatbot]).then(
        bot, chatbot, chatbot
    )

    example = gr.Examples(
        examples=[
            ["Is it correct to standardize your training and test data after you split or to split your dataset then standardize your training and test data?", None]
        ],
        inputs=[msg, upload]
    )

# Enable the queue when launching
demo.queue().launch(server_port=8100)

