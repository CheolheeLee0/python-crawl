import speech_recognition as sr
import os
from pydub import AudioSegment
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import logging

# 처리할 최대 파일 개수
MAX_FILES_TO_PROCESS = 1

def convert_mp3_to_wav(mp3_path):
    """MP3 파일을 WAV 파일로 변환"""
    audio = AudioSegment.from_mp3(mp3_path)
    wav_path = mp3_path.rsplit('.', 1)[0] + '.wav'
    audio.export(wav_path, format='wav')
    return wav_path

def convert_audio_to_text(audio_path):
    """오디오 파일을 텍스트로 변환"""
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-large-v3"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_id)
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        # 예시 오디오 사용
        result = pipe(
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
            return_timestamps=True,
            generate_kwargs={
                "language": "ko",
                "task": None
            }
        )
        
        return result["text"]
        
    except Exception as e:
        return f"에러 발생: {str(e)}"

def process_mp3_files():
    """downloads/videos 디렉토리의 모든 MP3 파일 처리"""
    # 로깅 설정
    logging.basicConfig(
        filename='server.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    directory = os.path.join('.', 'downloads', 'videos')
    
    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(directory, exist_ok=True)
    logging.info(f"작업 디렉토리 확인: {directory}")
    
    # 모든 MP3 파일 목록 가져오기
    mp3_files = [f for f in os.listdir(directory) if f.endswith('.mp3')]
    if not mp3_files:
        logging.warning(f"MP3 파일을 찾을 수 없습니다: {directory}")
        print(f"MP3 파일을 찾을 수 없습니다: {directory}")
        return
    
    # 처리할 파일 수 제한
    mp3_files = mp3_files[:MAX_FILES_TO_PROCESS]
    total_files = len(mp3_files)
    logging.info(f"총 처리할 파일 수: {total_files}")
    
    # 제한된 MP3 파일 처리
    for idx, filename in enumerate(mp3_files, 1):
        mp3_path = os.path.join(directory, filename)
        logging.info(f"파일 처리 시작: {filename} ({idx}/{total_files})")
        print(f"처리 중: {filename} ({idx}/{total_files})")
        
        try:
            # MP3를 WAV로 변환
            wav_path = convert_mp3_to_wav(mp3_path)
            logging.info(f"WAV 변환 완료: {wav_path}")
            
            # 음성을 텍스트로 변환
            text = convert_audio_to_text(wav_path)
            logging.info("음성-텍스트 변환 완료")
            
            # 결과를 텍스트 파일로 저장
            txt_filename = filename.rsplit('.', 1)[0] + '.txt'
            txt_path = os.path.join(directory, txt_filename)
            
            # 텍스트가 비어있지 않은 경우에만 저장
            if text and text.strip():
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                logging.info(f"텍스트 파일 저장 완료: {txt_filename}")
                print(f"변환 완료: {txt_filename}")
            else:
                logging.warning(f"변환된 텍스트가 비어있습니다: {filename}")
                print(f"변환된 텍스트가 비어있습니다: {filename}")
            
            # 임시 WAV 파일 삭제
            if os.path.exists(wav_path):
                os.remove(wav_path)
                logging.info(f"임시 WAV 파일 삭제 완료: {wav_path}")
                
        except Exception as e:
            error_msg = f"파일 처리 중 오류 발생: {filename}\n오류 내용: {str(e)}"
            logging.error(error_msg)
            print(error_msg)

if __name__ == "__main__":
    process_mp3_files()
