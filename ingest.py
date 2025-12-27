import os
import unicodedata
from pinecone import Pinecone
from google import genai

# 1. HARDCODED KEYS (Keep these secret!)
PINECONE_KEY = "pcsk_5Z6WFV_7oQoBHq752tQWBHA2VWdRPA7cYWCZgHyVeLBATX4iVnd899XA6N7XgeTRLFTCDK"
PINECONE_INDEX = "remcensus"
GEMINI_KEY = "AIzaSyBF7yZAfy-na9pO52yfGQIhnqpFNNsvRjM"

# 2. Initialize Clients
pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(PINECONE_INDEX)
genai_client = genai.Client(api_key=GEMINI_KEY)

def force_ascii(text):
    # Cleans "smart quotes" and other weird symbols to prevent crashes
    normalized = unicodedata.normalize('NFKD', text)
    return normalized.encode('ascii', 'ignore').decode('ascii')

def ingest_file(file_path):
    print(f"Processing: {file_path}...")
    
    # Read text
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except Exception as e:
        print(f"Skipping {file_path} due to read error: {e}")
        return

    # Clean text
    clean_text = force_ascii(raw_text)

    # Generate numbers (Embeddings)
    try:
        result = genai_client.models.embed_content(
            model="text-embedding-004",
            contents=clean_text
        )
        vector = result.embeddings[0].values

        # Upload to Pinecone
        index.upsert(
            vectors=[
                {
                    "id": file_path, 
                    "values": vector, 
                    "metadata": {"text": clean_text, "source": os.path.basename(file_path)}
                }
            ]
        )
        print(f" --> Success! Uploaded {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f" --> Failed to embed/upload {file_path}: {e}")

def batch_ingest(folder_name):
    # Loop through every file in the folder
    if not os.path.exists(folder_name):
        print(f"Error: Folder '{folder_name}' not found.")
        return

    files_found = 0
    for filename in os.listdir(folder_name):
        if filename.endswith(".txt"):
            files_found += 1
            full_path = os.path.join(folder_name, filename)
            ingest_file(full_path)
    
    if files_found == 0:
        print("No .txt files found in the folder.")
    else:
        print("\nBatch ingestion complete!")

if __name__ == "__main__":
    # Point this to your folder name
    batch_ingest("articles")