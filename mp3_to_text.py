import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging
from tqdm import tqdm
import sys
import os
import librosa
import glob

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
MAX_FILES_TO_PROCESS = 20  # 최대 처리 파일 개수 설정

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

def process_mp3_files():
    """downloads/videos 디렉토리의 mp3 파일들을 처리하는 함수"""
    try:
        # MP3 파일 경로 설정
        mp3_dir = "downloads/videos"
        mp3_files = glob.glob(os.path.join(mp3_dir, "*.mp3"))
        
        if not mp3_files:
            logger.warning(f"No MP3 files found in {mp3_dir}")
            return
        
        # 처리할 파일 수 제한
        mp3_files = mp3_files[:MAX_FILES_TO_PROCESS]
        logger.info(f"Found {len(mp3_files)} MP3 files to process")
        
        for mp3_file in mp3_files:
            try:
                # MP3 파일 로드
                logger.info(f"Processing file: {mp3_file}")
                audio_array, sampling_rate = librosa.load(mp3_file, sr=16000)
                
                # 음성을 텍스트로 변환
                transcribed_text = transcribe_audio({
                    'array': audio_array,
                    'sampling_rate': sampling_rate
                })
                
                # 결과를 txt 파일로 저장
                txt_file = os.path.splitext(mp3_file)[0] + '.txt'
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(transcribed_text)
                logger.info(f"Transcription saved to: {txt_file}")
                
            except Exception as e:
                logger.error(f"Error processing file {mp3_file}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error in process_mp3_files: {str(e)}")
        raise

# 메인 실행 부분 수정
if __name__ == "__main__":
    try:
        process_mp3_files()
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
