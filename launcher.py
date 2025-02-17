import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
import os

def launch_training():
    role = "YOUR_SAGEMAKER_ROLE_ARN"  # Replace with your SageMaker role ARN
    
    # Initialize SageMaker session
    session = sagemaker.Session()
    
    # Define hyperparameters
    hyperparameters = {
        'epochs': 3,
        'batch-size': 4,
        'learning-rate': 2e-5,
    }
    
    # Configure Hugging Face estimator
    huggingface_estimator = HuggingFace(
        entry_point='train.py',
        source_dir='.',
        instance_type='ml.g5.2xlarge',  # or 'ml.trn1.2xlarge' for Trainium
        instance_count=1,
        role=role,
        transformers_version='4.28.1',
        pytorch_version='2.0.0',
        py_version='py39',
        hyperparameters=hyperparameters,
        output_path='s3://outbound-ai/model-output'
    )
    
    # Define data channels
    data_channels = {
        'training': 's3://outbound-ai/training-data'
    }
    
    # Launch training job
    huggingface_estimator.fit(data_channels)

if __name__ == "__main__":
    launch_training()