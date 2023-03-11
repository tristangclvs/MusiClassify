import librosa
import librosa.display
import soundfile as sf
import os
from audio_converters import audio_to_wav
import simpleaudio as sa

genre: str = input("Enter the genre of the musics you want to scrape: ")

file_path = f"downloads\\{genre.capitalize()}\\"

sr = 22050
for file in os.listdir(file_path):
    abs_file_path = os.path.join(os.getcwd(), file_path, file)

    # print(abs_file_path)
    audio_path = audio_to_wav(abs_file_path)
    # print(audio)

    signal, sample_rate = librosa.load(audio_path, sr=sr)

    # Set the start and end times for trimming (in seconds)
    start_time = 0
    end_time = 30

    # Calculate the start and end samples for trimming
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Trim the audio file
    trimmed_audio = signal[start_sample:end_sample]

    # Export the trimmed audio to a file
    sf.write(f"{file.split('.')[0]}.wav", trimmed_audio, sr)
