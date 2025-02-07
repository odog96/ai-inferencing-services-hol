# clear pine cone db
import os
from pinecone import Pinecone, ServerlessSpec


## add this to env variables
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
print("Pinecone initialised")

pc.delete_index(name=PINECONE_INDEX)
print("Deleted Pinecone index")