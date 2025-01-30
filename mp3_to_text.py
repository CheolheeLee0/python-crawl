import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
import logging
from tqdm import tqdm
import sys

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
    torch.cuda.empty_cache()  # CUDA 캐시 정리
    logger.info(f"GPU found: {torch.cuda.get_device_name(0)}")
    logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

logger.info(f"Using device: {device}, dtype: {torch_dtype}")

# 모델 ID 설정
model_id = "openai/whisper-large-v3-turbo"
logger.info(f"Loading model: {model_id}")

# 전역 변수 선언
model = None
processor = None
pipe = None

def initialize_model():
    global model, processor, pipe
    
    if model is not None and processor is not None and pipe is not None:
        logger.info("Model already loaded")
        return

    # 모델 로드
    try:
        print("모델을 로드하는 중...")
        with tqdm(total=100, desc="모델 로딩", file=sys.stdout) as pbar:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True,
            )
            pbar.update(50)
            model.to(device)
            torch.cuda.empty_cache()  # CUDA 캐시 정리
            pbar.update(50)
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

def transcribe_audio(audio_input):
    global pipe
    
    if pipe is None:
        initialize_model()
    
    try:
        logger.info("Starting speech recognition...")
        print("\n음성을 텍스트로 변환하는 중...")
        
        # audio_input이 dictionary인지 확인하고 처리
        if isinstance(audio_input, dict):
            audio_array = audio_input['array']
            sampling_rate = audio_input['sampling_rate']
        else:
            audio_array = audio_input
            sampling_rate = 16000  # 기본 샘플링 레이트
        
        result = pipe(
            {"sampling_rate": sampling_rate, "array": audio_array}, 
            return_timestamps=True,
            chunk_length_s=30,
            stride_length_s=5
        )
        
        transcribed_text = result["text"]
        logger.info("Speech recognition completed")
        logger.info(f"Transcription result: {transcribed_text}")
        print("\n변환 결과:")
        print(transcribed_text)
        return transcribed_text
    except Exception as e:
        logger.error(f"Error during speech recognition: {str(e)}")
        raise

# 예시 실행
if __name__ == "__main__":
    try:
        logger.info("Loading dataset...")
        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        sample = dataset[0]["audio"]
        logger.info("Dataset loaded successfully")
        
        # 오디오 샘플 정보 출력
        print("\n=== 오디오 샘플 정보 ===")
        print(f"샘플링 레이트: {sample['sampling_rate']} Hz")
        print(f"오디오 길이: {len(sample['array'])} 샘플")
        print(f"재생 시간: {len(sample['array'])/sample['sampling_rate']:.2f} 초")
        
        # 음성 인식 실행
        transcribe_audio(sample)  # 전체 sample 딕셔너리 전달
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
