"""One-time smoke test. Delete after verifying."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. Pinecone reachable ---
from pinecone import Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
print("Pinecone OK:", index.describe_index_stats())

# --- 2. Bedrock invoke ---
import boto3
client = boto3.client("bedrock-runtime", region_name=os.environ["AWS_REGION"])
import json
response = client.invoke_model(
    modelId=os.environ["BEDROCK_MODEL_ID"],
    body=json.dumps({
        "schemaVersion": "messages-v1",
        "messages": [
            {
                "role": "user",
                "content": [{"text": "Reply with the single word: pong"}],
            }
        ],
    }),
)

payload = json.loads(response["body"].read())
print("Bedrock OK:", payload["output"]["message"]["content"][0]["text"])

# --- 3. Bedrock embeddings reachable ---
from langchain_aws import BedrockEmbeddings
embedder = BedrockEmbeddings(
    model_id=os.environ["BEDROCK_EMBEDDING_MODEL_ID"],
    region_name=os.environ["AWS_REGION"],
)
vec = embedder.embed_query("hello world")
print("Embedder OK:", len(vec))