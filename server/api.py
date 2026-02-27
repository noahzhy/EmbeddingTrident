"""
APIs for evaluating the performance of the model.
"""

import os
import time

import jax.numpy as jnp
import numpy as np
import yaml
import ray
from loguru import logger
from ray.data import ActorPoolStrategy

from preprocess import build_preprocess_batch, extract_image_size
from milvus_ops import MilvusInserter, build_index_and_load, create_collection_no_index
from triton_ops import TritonClient


# functions for copy files to make it as a triton model folder structure such as:
# model_repository/
#     model_name/
#         1/
#             model.onnx
def build_triton_model_folder(model_name, model_path, model_repository):
    model_folder = os.path.join(model_repository, model_name)
    version_folder = os.path.join(model_folder, "1")
    os.makedirs(version_folder, exist_ok=True)
    dest_model_path = os.path.join(version_folder, os.path.basename(model_path))
    if not os.path.exists(dest_model_path):
        os.symlink(os.path.abspath(model_path), dest_model_path)
    return dest_model_path


# funct to start a triton server with the given model repository
def start_triton_server(model_repository, triton_port=8000):
    # start triton server with the given model repository
    os.system(
        f"tritonserver --model-repository={model_repository} --http-port={triton_port} > triton_server.log 2>&1 &"
    )
    # wait for the server to start
    time.sleep(10)
    logger.info(f"Triton server started with model repository: {model_repository}")
    return triton_port


