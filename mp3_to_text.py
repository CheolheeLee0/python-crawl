import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import logging
from tqdm import tqdm
import sys

# 로깅 설정
logging.basicConfig(
    filename='server.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU 사용 여부 확인 및 로깅
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
logger.info(f"Using device: {device}, dtype: {torch_dtype}")

# 모델 ID 설정
model_id = "openai/whisper-large-v3"
logger.info(f"Loading model: {model_id}")

# 모델 로드
try:
    print("모델을 로드하는 중...")
    with tqdm(total=100, desc="모델 로딩", file=sys.stdout) as pbar:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True,
            force_download=True
        )
        pbar.update(50)  # 모델 다운로드 완료
        model.to(device)
        pbar.update(50)  # 모델 GPU 이동 완료
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# 프로세서 로드
try:
    processor = AutoProcessor.from_pretrained(model_id)
    logger.info("Processor loaded successfully")
except Exception as e:
    logger.error(f"Error loading processor: {str(e)}")
    raise

# 파이프라인 설정
try:
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    logger.info("Pipeline created successfully")
except Exception as e:
    logger.error(f"Error creating pipeline: {str(e)}")
    raise

# 데이터셋 로드
try:
    logger.info("Loading dataset...")
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]
    logger.info("Dataset loaded successfully")
except Exception as e:
    logger.error(f"Error loading dataset: {str(e)}")
    raise

# 음성 인식 실행
try:
    logger.info("Starting speech recognition...")
    print("\n음성을 텍스트로 변환하는 중...")
    
    def progress_callback(current_step, total_steps):
        if not hasattr(progress_callback, 'pbar'):
            progress_callback.pbar = tqdm(total=total_steps, desc="음성 인식")
        progress_callback.pbar.update(1)
        progress_callback.pbar.refresh()
    
    result = pipe(
        sample,
        generate_kwargs={'callback': progress_callback}
    )
    
    if hasattr(progress_callback, 'pbar'):
        progress_callback.pbar.close()
    
    logger.info("Speech recognition completed")
    print("\n변환 결과:")
    print(result["text"])
    logger.info(f"Transcription result: {result['text']}")
except Exception as e:
    logger.error(f"Error during speech recognition: {str(e)}")
    raise
