import os, sys, time, glob, re, io
from typing import List
from pathlib import Path

import loguru
from retrying import retry
from azure.storage.blob import BlobServiceClient, ContentSettings


# ENVIRONMENT VARIABLES
RETAIL_MLMODELS_CONNECTION_STRING = os.getenv("RETAIL_MLMODELS_CONNECTION_STRING")
RETAIL_PMS_CONNECTION_STRING = os.getenv("RETAIL_PMS_CONNECTION_STRING")


class BlobManager:
    def __init__(self, connection_string=None):
        if connection_string is None:
            raise ValueError("connection_string is not set")
        try:
            self.blob_client = BlobServiceClient.from_connection_string(connection_string)
        except Exception as e:
            print(e)
            self.blob_client = None
    
    def upload_blob(self, container_name, blob_name, data):
        blob_client = self.blob_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(data)
    
    @retry(wait_fixed=1000, stop_max_attempt_number=5)
    def download_blob(self, container_name, blob_name, dst_file):
        blob_client = self.blob_client.get_blob_client(container=container_name, blob=blob_name)
        dst_parent = os.path.dirname(dst_file)
        if dst_parent:
            os.makedirs(dst_parent, exist_ok=True)
        with open(dst_file, "wb") as f:
            data = blob_client.download_blob().readall()
            f.write(data)
    
    def delete_blob(self, container_name, blob_name):
        blob_client = self.blob_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.delete_blob()
    
    def list_blobs(self, container_name):
        container_client = self.blob_client.get_container_client(container_name)
        blobs = container_client.list_blobs()
        # for blob in blobs:
        #     print(blob.name)
        return [blob.name for blob in blobs]
    
    def create_container(self, container_name):
        container_client = self.blob_client.get_container_client(container_name)
        container_client.create_container()
    
    def delete_container(self, container_name):
        container_client = self.blob_client.get_container_client(container_name)
        container_client.delete_container()
    
    def list_containers(self):
        return [container.name for container in self.blob_client.list_containers()]
    
    ########### custom methods ###########
    def get_config_path(self, key_strings: List[str], container_name="ai-tool-csvs") -> List[str]:
        container_client = self.blob_client.get_container_client(container_name)
        unit_dirs = []
        for item in container_client.walk_blobs(delimiter="/"):
            if any(k in item.name for k in key_strings):
                unit_dirs.append(item.name)  # e.g. "27_CCTH-Unit/"
        
        for unit_dir in unit_dirs:
            prefix = f"{unit_dir}CSV/config/"
            
            for item in container_client.walk_blobs(name_starts_with=prefix, delimiter="/"):
                if any(k in item.name for k in key_strings):
                    ts_dir = item.name  # e.g. ".../20260317083337/"
                    for blob in container_client.list_blobs(name_starts_with=ts_dir):
                        if blob.name.endswith("config.json"):
                            return f"https://mlplatform.blob.core.chinacloudapi.cn/{container_name}/{blob.name}"
        
        return []

    def get_model_path(self, model_type:str, key_strings: List[str], container_name="ai-tool-models") -> List[str]:
        if model_type not in ["Object_Detection", "Classification", "Segmentation", "Unit_Sku_Detection"]:
            raise ValueError("model_type must be one of Object_Detection, Classification, Segmentation, Unit_Sku_Detection")

        container_client = self.blob_client.get_container_client(container_name)
        base_prefix = f"{model_type}/"

        # List all files under the model type path once, then filter in-memory.
        all_blob_names = [blob.name for blob in container_client.list_blobs(name_starts_with=base_prefix)]
        if not all_blob_names:
            return []

        # Match blobs that contain all key strings to locate the concrete model folder.
        matched_blob_names = [
            name for name in all_blob_names
            if all(key in name for key in key_strings)
        ]
        if not matched_blob_names:
            return []

        matched_prefixes = sorted(
            {
                name.rsplit("/", 1)[0] + "/"
                for name in matched_blob_names
                if "/" in name
            },
            key=len,
            reverse=True,
        )
        if not matched_prefixes:
            return []

        target_prefix = matched_prefixes[0]
        return [
            f"https://mlplatform.blob.core.chinacloudapi.cn/{container_name}/{name}"
            for name in all_blob_names
            if name.startswith(target_prefix) and not name.endswith("/")
        ]

    def download_model(self,
        model_type:str,
        key_strings: List[str],
        target_folder: str = "./downloaded_models",
        container_name="ai-tool-models"
    ):
        target_folder = os.path.abspath(target_folder)
        os.makedirs(target_folder, exist_ok=True)

        model_urls = self.get_model_path(model_type, key_strings, container_name)
        if not model_urls:
            return []

        local_files = []
        for url in model_urls:
            blob_name = url.split(f"{container_name}/")[1]
            dst_file = os.path.join(target_folder, blob_name.split("/")[-1])
            self.download_blob(container_name, blob_name, dst_file)
            local_files.append(dst_file)

        return local_files


def test_blob_manager(
        model_type="Object_Detection",
        model_name="CCTH-Unit",
        timestamp="20260317083337",
        target_folder="downloaded_models",
    ):
    blob_manager = BlobManager(RETAIL_MLMODELS_CONNECTION_STRING)
    config_path = blob_manager.get_config_path([model_name, timestamp])
    print("config_path:", config_path)

    model_files = blob_manager.get_model_path(model_type, [model_name, timestamp])
    print("model_files:", model_files)

    downloaded_files = blob_manager.download_model(
        model_type,
        [model_name, timestamp],
        target_folder=f"{target_folder}/{model_name}_{timestamp}",
    )
    print("downloaded_files:", downloaded_files)
    # ready to run 
    print(f"""
        python utils/generate_unit_triton.py {target_folder}/{model_name}_{timestamp} trt_models \\
        --model-name {model_name} \\
        --onnx-model-name _{model_name} \\
        --postprocess-model-name unit_postprocess

        bash scripts/start_triton.sh trt_models/
    """)



if __name__ == '__main__':
    test_blob_manager(
        model_type="Object_Detection",
        model_name="CCTH-Unit",
        timestamp="20260317083337",
        target_folder="./test_models",
    )