import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import logging
from tqdm import tqdm
import sys

# CUDA 확인 부분을 try-except로 감싸서 안전하게 처리
try:
    print("CUDA is available:", torch.cuda.is_available())
    print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "Not available")
except:
    print("CUDA check failed")

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# GPU 사용 여부 확인 및 로깅
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

if torch.cuda.is_available():
    try:
        torch.cuda.empty_cache()  # CUDA 캐시 정리
        
        # GPU 기본 정보
        gpu_name = torch.cuda.get_device_name(0)
        gpu_properties = torch.cuda.get_device_properties(0)
        
        # 메모리 정보
        total_memory = gpu_properties.total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        
        # GPU 상세 정보 로깅
        logger.info("=== GPU Information ===")
        logger.info(f"GPU Model: {gpu_name}")
        logger.info(f"Compute Capability: {gpu_properties.major}.{gpu_properties.minor}")
        logger.info(f"Total GPU Memory: {total_memory:.2f} GB")
        logger.info(f"Currently Allocated Memory: {allocated_memory:.2f} GB")
        logger.info(f"Currently Reserved Memory: {reserved_memory:.2f} GB")
        logger.info(f"Multi Processor Count: {gpu_properties.multi_processor_count}")
        logger.info(f"Max Threads Per Block: {gpu_properties.max_threads_per_block}")
        logger.info(f"Max Threads Per MP: {gpu_properties.max_threads_per_multi_processor}")
        
        # CUDA 버전 정보
        logger.info("=== CUDA Information ===")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        logger.info(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        
    except Exception as e:
        logger.error(f"Error accessing GPU properties: {str(e)}")
else:
    logger.warning("GPU is not available. Using CPU instead.")

logger.info(f"Using device: {device}, dtype: {torch_dtype}")
