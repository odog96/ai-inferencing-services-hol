import os
import gradio as gr
import cmlapi
import pinecone
from pinecone import Pinecone, ServerlessSpec
from typing import Any, Union, Optional
from pydantic import BaseModel
#import tensorflow as tf
from sentence_transformers import SentenceTransformer
import requests
import json
import time
from typing import Optional

from openai import OpenAI
import httpx
from typing import List, Dict, Generator

# import boto3
# from botocore.config import Config
# import chromadb
# from chromadb.utils import embedding_functions

from huggingface_hub import hf_hub_download

EMBEDDING_MODEL_REPO = "sentence-transformers/all-mpnet-base-v2"


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')

print("initialising Pinecone connection...")
pc = Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone initialised")

print(f"Getting '{PINECONE_INDEX}' as object...")
index = pc.Index(PINECONE_INDEX)
print("Success")

rerank_url = "https://ai-inference.ainf-cdp.vayb-xokg.cloudera.site/namespaces/serving-default/endpoints/mistral-4b-rerank-a10g/v1/ranking"
rerank_model_name = "nvidia/nv-rerankqa-mistral-4b-v3"

# Load API key
OPENAI_API_KEY = json.load(open("/tmp/jwt"))["access_token"]

# Get latest statistics from index
current_collection_stats = index.describe_index_stats()
print('Total number of embeddings in Pinecone index is {}.'.format(current_collection_stats.get('total_vector_count')))

class ChatClient:
    def __init__(self, model_name: str, base_url: str, temperature: float, token_count: int):
        self.model_name = model_name  # Store as instance variable
        self.base_url = base_url      # Store as instance variable
        self.token_count = token_count
        self.temperature = temperature
        
        # Set up HTTP client
        print('model_name',model_name)
        if "CUSTOM_CA_STORE" not in os.environ:
            http_client = httpx.Client()
        else:
            http_client = httpx.Client(verify=os.environ["CUSTOM_CA_STORE"])
            
        # Load API key
        OPENAI_API_KEY = json.load(open("/tmp/jwt"))["access_token"]
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url= base_url,
            api_key=OPENAI_API_KEY,
            http_client=http_client,
        )
        
        self.conversation_history: List[Dict[str, str]] = []
        
    def chat(self, message: str, stream: bool = True) -> str:
        """
        Send a message to the chat model and get the response.
        
        Args:
            message: The message to send to the model
            stream: Whether to stream the response or return it all at once
            
        Returns:
            The complete response as a string
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": message})
        
        if stream:
            partial_message = ""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                  temperature=self.temperature,
              max_tokens=self.token_count,
                stream=True,
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    partial_message += content
                    print(content, end='', flush=True)
            
            print()  # New line after response is complete
            # Add complete response to history
            self.conversation_history.append({"role": "assistant", "content": partial_message})
            return partial_message
            
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.conversation_history,
                temperature=self.temperature,
                max_tokens = self.token_count,
                stream=False,
            )
            complete_response = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": complete_response})
            return complete_response
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []    
   
def main():
    # Configure gradio QA app 
    print("Configuring gradio app")

    DESC = "This AI-powered assistant showcases the flexibility of Cloudera Machine Learning to work with 3rd party solutions for LLMs and Vector Databases, as well as internally hosted models and vector DBs. The prototype does not yet implement chat history and session context - every prompt is treated as a brand new one."
    
    # Create the Gradio Interface
    demo = gr.ChatInterface(
        fn=get_responses, 
        title="AI Inference Rag Pipeline with re rank model",
        description = DESC,
        additional_inputs=[gr.Radio(['Llama-3.1-8b-instruct', 'Deepseek r1 distill'], label="Select Foundational Model", value="Llama-3.1-8b-instruct"), 
                           gr.Slider(minimum=0.0, maximum=2.0, step=0.25, value=0.5, label="Select Temperature (Randomness of Response)"),
                           gr.Radio(["64", "128", "256", "512", "1028"], label="Select Number of Tokens (Length of Response)", value="256"),gr.Checkbox(label="Enable Reranking", value=False) ],
        retry_btn = None,
        undo_btn = None,
        clear_btn = None,
        autofocus = True
        )

    # Launch gradio app
    print("Launching gradio app")
    demo.launch(share=True,   
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_READONLY_PORT'))
               )
    print("Gradio app ready")
    
# Helper function for generating responses for the QA app
def get_responses(message, history, model, temperature, token_count, enable_rerank):
    token_count = int(token_count) 
        # Vector search the index
    context_chunk, source, score = get_nearest_chunk_from_pinecone_vectordb(index, message, enable_rerank=enable_rerank
    )

    print('source',source)
    #print('context chunk',context_chunk)
    
    # Call CML hosted model
    response = get_llm_response_with_context(message, context_chunk, temperature, model, token_count)
    
    # Add reference to specific document in the response
    response = f"{response}\n\n For additional info see: {url_from_source(source)}"
    
    # Stream output to UI
    for i in range(len(response)):
        time.sleep(0.02)
        yield response[:i+1]

def url_from_source(source):
    url = source.replace('/home/cdsw/data/https:/', 'https://').replace('.txt', '.html')
    return f"[Reference 1]({url})"
    
def get_nearest_chunk_from_pinecone_vectordb(index, question, enable_rerank=True):
    retriever = SentenceTransformer(EMBEDDING_MODEL_REPO)
    xq = retriever.encode([question]).tolist()
    xc = index.query(vector=xq, top_k=10, include_metadata=True)  # Back to 10

    # # Print original vector rankings
    # print("\nOriginal Vector Rankings:")
    # for i, match in enumerate(xc['matches']):
    #     print(f"{i+1}. Score: {match['score']:.4f} - Path: {match['metadata']['file_path']}")
    
    if enable_rerank:
        passages = [load_context_chunk_from_data(match['metadata']['file_path']) 
                   for match in xc['matches']]
        
        # Helper function to truncate text
        def truncate_text(text, max_words=50):  # 50 words is roughly 65-75 tokens
            words = text.split()
            return ' '.join(words[:max_words])
        
        formatted_passages = [{"text": truncate_text(passage)} for passage in passages]
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Accept": "application/json",
        }
        
        payload = {
            "model": rerank_model_name,
            "query": {
                "text": question
            },
            "passages": formatted_passages
        }
        
        try:
            print("\nFirst passage (truncated):", formatted_passages[0]["text"][:100], "...")
            print("Total passages:", len(formatted_passages))
            print("Query:", question)
            
            session = requests.Session()
            response = session.post(rerank_url, headers=headers, json=payload)
            
            if response.status_code != 200:
                print("Error status:", response.status_code)
                print("Error response:", response.text[:200])
            
            response.raise_for_status()
            rankings = response.json().get('rankings', [])
            
            if not rankings:
                print("No rankings returned, falling back to original ranking")
                match = xc['matches'][0]
                return (load_context_chunk_from_data(match['metadata']['file_path']),
                        match['metadata']['file_path'],
                        match['score'])
            
            best_idx = rankings[0]['index']
            best_match = xc['matches'][best_idx]
            best_score = rankings[0]['logit']
            
            return (load_context_chunk_from_data(best_match['metadata']['file_path']), 
                    best_match['metadata']['file_path'], 
                    best_score)
                    
        except Exception as e:
            print(f"Reranking error: {str(e)}")
            match = xc['matches'][0]
            return (load_context_chunk_from_data(match['metadata']['file_path']),
                    match['metadata']['file_path'],
                    match['score'])

    match = xc['matches'][0]
    return (load_context_chunk_from_data(match['metadata']['file_path']),
            match['metadata']['file_path'],
            match['score'])
# Get embeddings for a user question and query Pinecone vector DB for nearest knowledge base chunk
# def get_nearest_chunk_from_pinecone_vectordb(index, question, enable_rerank=True):
#     retriever = SentenceTransformer(EMBEDDING_MODEL_REPO)
#     xq = retriever.encode([question]).tolist()
#     xc = index.query(vector=xq, top_k=10, include_metadata=True)

#     if enable_rerank:
#         passages = [load_context_chunk_from_data(match['metadata']['file_path']) 
#                    for match in xc['matches']]
                   
#         # Format for NVIDIA reranker
#         rerank_request = {
#             "input": {
#                 "query": question,
#                 "passages": passages
#             }
#         }
        
#         rerank_client = OpenAI(
#             base_url=rerank_url,
#             api_key=OPENAI_API_KEY
#         )
#         print('client created')
        
#         try:
#             completion = rerank_client.chat.completions.create(
#                 model=rerank_model_name,
#                 messages=[{
#                     "role": "user",
#                     "content": json.dumps(rerank_request)
#                 }],
#                 temperature=0.0
#             )
#             print('completion received')
            
#             # Parse the response
#             response = json.loads(completion.choices[0].message.content)
#             reranked_scores = response["scores"] if "scores" in response else []
            
#             if not reranked_scores:
#                 print("No scores returned, falling back to original ranking")
#                 match = xc['matches'][0]
#                 return (load_context_chunk_from_data(match['metadata']['file_path']),
#                         match['metadata']['file_path'],
#                         match['score'])
            
#             # Get best match after reranking
#             best_idx = max(range(len(reranked_scores)), key=reranked_scores.__getitem__)
#             best_match = xc['matches'][best_idx]
            
#             return (load_context_chunk_from_data(best_match['metadata']['file_path']), 
#                     best_match['metadata']['file_path'], 
#                     reranked_scores[best_idx])
                    
#         except Exception as e:
#             print(f"Reranking error: {str(e)}")
#             # Fallback to original behavior
#             match = xc['matches'][0]
#             return (load_context_chunk_from_data(match['metadata']['file_path']),
#                     match['metadata']['file_path'],
#                     match['score'])

#     # Original behavior
#     match = xc['matches'][0]
#     return (load_context_chunk_from_data(match['metadata']['file_path']),
#             match['metadata']['file_path'],
#             match['score'])


# Return the Knowledge Base doc based on Knowledge Base ID (relative file path)
def load_context_chunk_from_data(id_path):
    with open(id_path, "r") as f: # Open file in read mode
        return f.read()
    
# Pass through user input to LLM model with enhanced prompt and stop tokens
def get_llm_response_with_context(question, context, temperature, model, token_count):
    
    llama_sys = f"<s>[INST]You are a helpful, respectful and honest assistant. If you are unsure about an answer, truthfully say \"I don't know\"."
    
    if context == "":
        # Following LLama's spec for prompt engineering
        llama_inst = f"Please answer the user question.[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} \n [INST] {question} [/INST]"
    else:
        # Add context to the question
        llama_inst = f"Anser the user's question based on the folloing information:\n {context}[/INST]</s>"
        question_and_context = f"{llama_sys} {llama_inst} \n[INST] {question} [/INST]"

    if model == 'Llama-3.1-8b-instruct':
        model_name = "meta/llama-3.1-8b-instruct"
        base_url = "https://ai-inference.ainf-cdp.vayb-xokg.cloudera.site/namespaces/serving-default/endpoints/test-model-llama-8b-v2/v1"
        try:
            llama_chat_client = ChatClient(model_name, base_url, temperature, token_count)
    
            print('chat client created')
            
            # For streaming responses (will print as it receives chunks):
            response = llama_chat_client.chat(question_and_context, stream=True)
                       
            return response
        
        except Exception as e:
            print(e)
            return e
        
    else:
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        base_url = "https://ai-inference.ainf-cdp.vayb-xokg.cloudera.site/namespaces/serving-default/endpoints/deepseek-r1-distill-llama-8b/openai/v1"      

        try:
            ds_chat_client = ChatClient(model_name, base_url, temperature, token_count)
    
            print('chat client created')
            
            # For streaming responses (will print as it receives chunks):
            response = ds_chat_client.chat(question_and_context, stream=True)
                       
            return response
        
        except Exception as e:
            print(e)
            return e
    # try:
    #     chat_client = ChatClient(model_name, base_url, temperature, token_count)

    #     print('chat client created')
        
    #     # For streaming responses (will print as it receives chunks):
    #     response = chat_client.chat(question_and_context, stream=True)
                   
    #     return response
        
    # except Exception as e:
    #     print(e)
    #     return e


if __name__ == "__main__":
    main()
