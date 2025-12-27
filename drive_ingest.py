import warnings, logging, unicodedata, os, io, time
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pinecone import Pinecone
import google.generativeai as genai 
import pypdf

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)

print("✅ 'Remcensus Ingest System Online...")

PINECONE_KEY = "pcsk_5Z6WFV_7oQoBHq752tQWBHA2VWdRPA7cYWCZgHyVeLBATX4iVnd899XA6N7XgeTRLFTCDK"
PINECONE_INDEX = "remcensus"
GEMINI_KEY = "AIzaSyBF7yZAfy-na9pO52yfGQIhnqpFNNsvRjM"
ROOT_FOLDER_ID = "10B8EsEQ2TlzQP5ADD43TcDSs_xp3plj9" 
SERVICE_ACCOUNT_FILE = "service_account.json"

pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(PINECONE_INDEX)
genai.configure(api_key=GEMINI_KEY)

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=['https://www.googleapis.com/auth/drive']
)
drive_service = build('drive', 'v3', credentials=creds)

# --- HELPER FUNCTIONS ---
def sanitize_text(text):
    if not text: return ""
    replacements = {'\u2018':"'", '\u2019':"'", '\u201c':'"', '\u201d':'"', '\u2013':"-", '\u25cf':"*", '\u2022':"*"}
    for k, v in replacements.items(): text = text.replace(k, v)
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

def chunk_text(text, chunk_size=8000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def get_text_from_pdf(content):
    try:
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
    except: return ""

def process_file(file_id, name, mime):
    # 1. DOWNLOAD
    fh = io.BytesIO()
    request = None
    is_doc_or_sheet = False
    
    if mime == 'application/vnd.google-apps.document':
        request = drive_service.files().export_media(fileId=file_id, mimeType='text/plain')
        is_doc_or_sheet = True
    elif mime == 'application/vnd.google-apps.spreadsheet':
        request = drive_service.files().export_media(fileId=file_id, mimeType='text/csv')
        is_doc_or_sheet = True
    elif 'pdf' in mime or 'plain' in mime:
        request = drive_service.files().get_media(fileId=file_id)
    else: return 

    print(f"   📥 Downloading: {name}...")
    try:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done: _, done = downloader.next_chunk()
    except Exception as e:
        print(f"   ❌ Download Error: {e}")
        return

    # 2. EXTRACT
    content = fh.getvalue()
    raw = ""
    try:
        if is_doc_or_sheet or 'plain' in mime: raw = content.decode("utf-8")
        elif 'pdf' in mime: raw = get_text_from_pdf(content)
    except: return

    # 3. SANITIZE (The Important Part)
    clean_text = sanitize_text(raw)
    safe_name = sanitize_text(name) # Fixes metadata crash
    safe_id = safe_name.replace(" ", "_") # Fixes ID crash
    
    if not clean_text: return

    # 4. CHUNK & UPLOAD
    chunks = chunk_text(clean_text)
    print(f"   🧠 Processing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        try:
            vec = genai.embed_content(model="models/text-embedding-004", content=chunk)['embedding']
            index.upsert(vectors=[{
                "id": f"{safe_id}_part_{i}", 
                "values": vec, 
                "metadata": {"text": chunk, "source": safe_name}
            }])
        except Exception as e: print(f"      ❌ Chunk failed: {e}")
        time.sleep(0.5)
    print(f"   ✅ Finished: {name}")

def traverse_folder(folder_id):
    try:
        res = drive_service.files().list(q=f"'{folder_id}' in parents and trashed=false", fields="files(id, name, mimeType)").execute()
        for f in res.get('files', []):
            if f['mimeType'] == 'application/vnd.google-apps.folder':
                print(f"\n📂 Entering: {f['name']}")
                traverse_folder(f['id'])
            else:
                process_file(f['id'], f['name'], f['mimeType'])
    except Exception as e: print(f"Skipping folder: {e}")

if __name__ == "__main__":
    traverse_folder(ROOT_FOLDER_ID)
    print("\n🎉 All Done!")