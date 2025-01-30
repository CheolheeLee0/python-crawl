import os
from pydub import AudioSegment

def convert_wav_to_m4a(input_path, output_path):
    # WAV 파일 로드
    audio = AudioSegment.from_wav(input_path)
    
    # M4A로 내보내기 (사실상 AAC 코덱 사용)
    audio.export(output_path, format="ipod", codec="aac")

def recursive_convert(input_folder, output_folder):
    # 출력 폴더가 없으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 모든 하위 폴더를 포함하여 WAV 파일 처리
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".wav"):
                # 입력 파일의 전체 경로
                input_path = os.path.join(root, filename)
                
                # 출력 파일의 경로 생성
                relative_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(output_folder, relative_path, f"{os.path.splitext(filename)[0]}.m4a")
                
                # 출력 디렉토리 생성 (없는 경우)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                print(f"Converting {input_path} to M4A...")
                convert_wav_to_m4a(input_path, output_path)
                print(f"Converted: {output_path}")

if __name__ == "__main__":
    input_folder = "./downloads"  # WAV 파일이 있는 최상위 폴더 경로
    output_folder = "./downloads/compressed"  # 변환된 M4A 파일을 저장할 최상위 폴더 경로
    
    recursive_convert(input_folder, output_folder)
    print("Conversion complete!")