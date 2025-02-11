from openai import OpenAI
import gradio as gr
import os
import httpx

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


def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=os.environ["OPENAI_MODEL_NAME"],
        messages=history_openai_format,
        stream=True,
    )

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message


gr.ChatInterface(predict).launch()