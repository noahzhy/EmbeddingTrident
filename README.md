# Auto Model Evaluation Platform

## Project Overview
This project is designed to automate the evaluation of machine learning models, specifically focusing on object detection models. It provides a framework for downloading models, generating Triton Inference Server configurations, and visualizing results.

## Step-by-Step Guide

1. Download model relate files from blob via given model info { model_type, model_name, timestamp }.
2. Run script utils/generate_unit_triton.py if model type is Object_Detection, it will generate a model which follows triton infer server structure.
3. Run script scripts/start_triton.sh to start all models user want into triton server.
4. Start ray service and set unit and sku model name.
