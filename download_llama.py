import io
import json
import os
import urllib.request
import zipfile

print("Fetching latest llama.cpp release...")
req = urllib.request.Request("https://api.github.com/repos/ggerganov/llama.cpp/releases/latest")
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())

llama_url = None
cudart_url = None

for asset in data.get("assets", []):
    name = asset["name"]
    if "cudart-llama-bin-win-cuda-12.4-x64.zip" in name:
        cudart_url = asset["browser_download_url"]
    elif "-bin-win-cuda-12.4-x64.zip" in name and not name.startswith("cudart-"):
        llama_url = asset["browser_download_url"]

if not llama_url:
    print("Could not find CUDA Windows build.")
    exit(1)


def download_and_extract(url, out_dir):
    print(f"Downloading from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        zip_data = response.read()
    print("Extracting...")
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_data)) as z:
        for file_info in z.infolist():
            if file_info.filename.endswith(".exe") or file_info.filename.endswith(".dll"):
                file_info.filename = os.path.basename(file_info.filename)
                z.extract(file_info, out_dir)


download_and_extract(llama_url, "local_llm")
if cudart_url:
    download_and_extract(cudart_url, "local_llm")

print("Done! Extracted llama-server.exe and DLLs to local_llm/")
