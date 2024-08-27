from dotenv import load_dotenv
load_dotenv()
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import os
import json

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index exists and create it if it doesn't
index_name = "rag"
try:
    # Try to get the index
    index = pc.Index(index_name)
    print(f"Index '{index_name}' already exists. Skipping creation.")
except Exception as e:
    # If the index doesn't exist, create it
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created index: {index_name}")
    index = pc.Index(index_name)
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to reviews.json in the same directory as the script
reviews_path = os.path.join(current_dir, "reviews.json")

# Load the review data
try:
    with open(reviews_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: The file {reviews_path} was not found.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {current_dir}")
    raise


processed_data = []
client = OpenAI()

# Create embeddings for each review
for review in data["professor_reviews"]:  # Access the "professor_reviews" key
    try:
        response = client.embeddings.create(
            input=review['review'], model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        processed_data.append(
            {
                "values": embedding,
                "id": review["professor"],
                "metadata": {
                    "review": review["review"],
                    "subject": review["subject"],
                    "stars": review["stars"],
                }
            }
        )
    except Exception as e:
        print(f"Error processing review for {review['professor']}: {e}")

# Insert the embeddings into the Pinecone index
index = pc.Index(index_name)
upsert_response = index.upsert(
    vectors=processed_data,
    namespace="ns1",
)
print(f"Upserted count: {upsert_response['upserted_count']}")

# Print index statistics
print(index.describe_index_stats())