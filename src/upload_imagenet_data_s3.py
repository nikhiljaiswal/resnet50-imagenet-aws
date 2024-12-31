import os
import boto3
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
from boto3.s3.transfer import TransferConfig

S3_BUCKET_NAME = "imagenet-dataset-nikhil"

# Configure multipart upload
GB = 1024 ** 3
transfer_config = TransferConfig(
    multipart_threshold=1 * GB,
    max_concurrency=10,
    multipart_chunksize=1 * GB,
    use_threads=True
)

# Thread-safe counter for progress
class ProgressCounter:
    def __init__(self):
        self._counter = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._counter += 1
            return self._counter

def upload_file(args):
    file_path, s3_key, counter, pbar, total_files = args
    try:
        s3_client = boto3.client('s3')
        s3_client.upload_file(
            file_path, 
            S3_BUCKET_NAME, 
            s3_key,
            Config=transfer_config
        )
        uploaded = counter.increment()
        pbar.update(1)
        pbar.set_postfix({
            'uploaded': f"{uploaded}/{total_files}",
            'current_file': os.path.basename(file_path)
        })
        return True
    except Exception as e:
        print(f"\nError uploading {file_path}: {str(e)}")
        return False

def upload_to_s3():
    print("Preparing for S3 upload...")
    s3_client = boto3.client("s3")
    
    # Base directory to start from
    BASE_DIR = "ILSVRC/Data/CLS-LOC"
    
    # Collect all files first
    upload_tasks = []
    total_files = 0
    for root, _, files in os.walk(BASE_DIR):
        if "test" in root.split(os.path.sep):
            continue
        for file in files:
            file_path = os.path.join(root, file)
            s3_key = os.path.join("imagenet", file_path)
            s3_key = s3_key.replace("\\", "/")
            upload_tasks.append((file_path, s3_key))
            total_files += 1
    
    print(f"Found {total_files} files to upload")
    
    # Create progress bar and counter
    pbar = tqdm(total=total_files, desc="Uploading files")
    counter = ProgressCounter()
    
    # Prepare arguments for upload function
    upload_args = [
        (file_path, s3_key, counter, pbar, total_files)
        for file_path, s3_key in upload_tasks
    ]
    
    # Use ThreadPoolExecutor for parallel uploads
    successful_uploads = 0
    with ThreadPoolExecutor(max_workers=48) as executor:
        results = list(executor.map(upload_file, upload_args))
        successful_uploads = sum(1 for r in results if r)
    
    pbar.close()
    print(f"\nUpload complete! Successfully uploaded {successful_uploads}/{total_files} files")

# Main script
if __name__ == "__main__":
    upload_to_s3()
