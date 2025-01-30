import torch
import transformers
import datasets
import accelerate
import librosa
import soundfile
import numpy as np

def check_versions():
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"Transformers 버전: {transformers.__version__}")
    print(f"Datasets 버전: {datasets.__version__}")
    print(f"Accelerate 버전: {accelerate.__version__}")
    print(f"Librosa 버전: {librosa.__version__}")
    print(f"Soundfile 버전: {soundfile.__version__}")
    print(f"Numpy 버전: {np.__version__}")

if __name__ == "__main__":
    check_versions()