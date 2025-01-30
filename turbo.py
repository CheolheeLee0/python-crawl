import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
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

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

# 오디오 샘플 정보 출력
print("\n=== Audio Sample Info ===")
print(f"Sample duration: {sample['array'].shape[0] / sample['sampling_rate']:.2f} seconds")
print(f"Sampling rate: {sample['sampling_rate']} Hz")

# 오디오 데이터 전처리
audio_array = sample['array']
sampling_rate = sample['sampling_rate']

# 파이프라인 실행 (상세 로깅 활성화)
print("\n=== Processing Audio ===")
result = pipe(
    {"sampling_rate": sampling_rate, "array": audio_array}, 
    return_timestamps=True,
    chunk_length_s=30,  # 30초 단위로 처리
    stride_length_s=5   # 5초 오버랩
)

# 결과 상세 출력
print("\n=== Recognition Results ===")
print(f"Complete Text: {result['text']}")

if 'chunks' in result:
    print("\nText Chunks with Timestamps:")
    for chunk in result['chunks']:
        start = chunk['timestamp'][0]
        end = chunk['timestamp'][1]
        text = chunk['text']
        print(f"[{start:.2f}s -> {end:.2f}s] {text}")
