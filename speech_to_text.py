import openai
import os

# Set up the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Open the audio file
audio_file = open("./downloads/starterstory/He Built A $2.5M_Year Business In 2 Years.wav", "rb")

# Transcribe the audio
transcript = openai.Audio.transcribe("whisper-1", audio_file)

print(transcript["text"])


# def process_wav_files(directory):
#     results = []
#     for filename in os.listdir(directory):
#         if filename.endswith(".wav"):
#             file_path = os.path.join(directory, filename)
#             text = transcribe_wav(file_path)
#             results.append((filename, text))
#     return results

# # Specify the directory containing .wav files
# directory = "./downloads/starterstory"

# # Process all .wav files in the directory
# transcriptions = process_wav_files(directory)

# # Save results to a CSV file
# with open("transcriptions.csv", "w", newline='', encoding='utf-8') as output_file:
#     csv_writer = csv.writer(output_file)
#     csv_writer.writerow(["Filename", "Transcription"])  # Write header
#     for filename, text in transcriptions:
#         csv_writer.writerow([filename, text])

# print("Transcriptions saved to transcriptions.csv")