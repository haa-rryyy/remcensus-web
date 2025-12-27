from google import genai

# Setup the client
client = genai.Client(api_key="AIzaSyBF7yZAfy-na9pO52yfGQIhnqpFNNsvRjM")

print("Checking available models...")
try:
    # Just print every model name it finds
    for model in client.models.list():
        print(model.name)
            
except Exception as e:
    print(f"Error: {e}")