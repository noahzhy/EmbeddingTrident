import os, sys, time, glob, re, io, json, argparse
from typing import List
from pathlib import Path

import loguru
from retrying import retry
from azure.storage.blob import BlobServiceClient, ContentSettings


# ENVIRONMENT VARIABLES
RETAIL_MLMODELS_CONNECTION_STRING = os.getenv("RETAIL_MLMODELS_CONNECTION_STRING")
RETAIL_PMS_CONNECTION_STRING = os.getenv("RETAIL_PMS_CONNECTION_STRING")


def _get_common_dir_prefix(blob_names: List[str]) -> str:
    if not blob_names:
        raise ValueError("blob_names is empty")

    common_prefix = os.path.commonprefix(blob_names)
    if common_prefix.endswith("/"):
        return common_prefix

    if "/" not in common_prefix:
        return ""

    return common_prefix.rsplit("/", 1)[0] + "/"


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


    def download_model_folder(
        self,
        model_type: str,
        key_strings: List[str],
        local_dir: str,
        container_name="ai-tool-models"
    ) -> str:
        if model_type not in ["Object_Detection", "Classification", "Segmentation", "Unit_Sku_Detection"]:
            raise ValueError("invalid model_type")

        container_client = self.blob_client.get_container_client(container_name)
        base_prefix = f"{model_type}/"

        # 1. 单次扫描找匹配项，避免缓存整个 model_type 下的所有 blob
        has_blobs = False
        matched_blob_names = []
        for blob in container_client.list_blobs(name_starts_with=base_prefix):
            has_blobs = True
            if all(key in blob.name for key in key_strings):
                matched_blob_names.append(blob.name)

        if not has_blobs:
            raise ValueError("No blobs found")

        # 2. 找匹配的 blobs
        if not matched_blob_names:
            raise ValueError("No matched model found")

        # 3. 找公共父目录，确保同级目录（如 model/ 与 labelmap/）都会被下载
        target_prefix = _get_common_dir_prefix(matched_blob_names)
        if not target_prefix:
            raise ValueError("Failed to resolve model folder")

        # 4. 下载整个 folder
        download_root = os.path.join(local_dir, target_prefix)
        os.makedirs(download_root, exist_ok=True)

        for blob in container_client.list_blobs(name_starts_with=target_prefix):
            blob_name = blob.name
            if blob_name.endswith("/"):
                continue

            # 本地路径（保持目录结构）
            relative_path = blob_name[len(target_prefix):]
            local_path = os.path.join(download_root, relative_path)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            blob_client = container_client.get_blob_client(blob_name)

            with open(local_path, "wb") as f:
                stream = blob_client.download_blob()
                f.write(stream.readall())

        return download_root

    # def download_model(self,
    #     model_type:str,
    #     key_strings: List[str],
    #     target_folder: str = "./downloaded_models",
    #     container_name="ai-tool-models"
    # ):
    #     target_folder = os.path.abspath(target_folder)
    #     os.makedirs(target_folder, exist_ok=True)

    #     model_urls = self.get_model_path(model_type, key_strings, container_name)
    #     if not model_urls:
    #         return []

    #     local_files = []
    #     for url in model_urls:
    #         blob_name = url.split(f"{container_name}/")[1]
    #         dst_file = os.path.join(target_folder, blob_name.split("/")[-1])
    #         self.download_blob(container_name, blob_name, dst_file)
    #         local_files.append(dst_file)

    #     return local_files


# def test_blob_manager(
def download_and_prepare_model(
        model_type="Object_Detection",
        model_name="CCTH-Unit",
        timestamp="20260317083337",
        target_folder="downloaded_models",
    ):
    blob_manager = BlobManager(RETAIL_MLMODELS_CONNECTION_STRING)
    config_path = blob_manager.get_config_path([model_name, timestamp])
    print("config_path:", config_path)

    model_files = blob_manager.download_model_folder(model_type, [model_name, timestamp], target_folder)
    print("model_files:", model_files)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Download model from Azure Blob Storage and prepare for Triton")
    parser.add_argument("--model-type", type=str, default="Object_Detection", help="Type of the model to download (e.g. Object_Detection, Classification, Segmentation, Unit_Sku_Detection)")
    parser.add_argument("--model-name", type=str, default="CCTH-Unit", help="Name of the model to download")
    parser.add_argument("--timestamp", type=str, default="20260317083337", help="Timestamp to identify the model version")
    parser.add_argument("--target-folder", type=str, default="./downloaded_models", help="Local folder to save the downloaded model")
    args = parser.parse_args()

    download_and_prepare_model(
        model_type=args.model_type,
        model_name=args.model_name,
        timestamp=args.timestamp,
        target_folder=args.target_folder,
    )
