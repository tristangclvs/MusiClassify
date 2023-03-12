import librosa
import librosa.display
import soundfile as sf
import os
from audio_converters import audio_to_wav
import simpleaudio as sa


def trim_audio(abs_file_path, start_time, end_time, write=False, sr=22050):
    audio_path = audio_to_wav(abs_file_path)
    signal, sample_rate = librosa.load(audio_path, sr=sr)
    # Calculate the start and end samples for trimming
    start_sample: int = int(start_time * sr)
    end_sample: int = int(end_time * sr)

    # Trim the audio file
    trimmed_audio = signal[start_sample:end_sample]

    if write:
        # Export the trimmed audio to a file
        sf.write(f"{file.split('.')[0]}.wav", trimmed_audio, sr)

    return trimmed_audio


# ======================================================================================
genre: str = input("Enter the genre of the musics you want to scrape: ")

file_path: str = f"downloads\\{genre.capitalize()}\\"

for file in os.listdir(file_path):
    abs_file_path: str = os.path.join(os.getcwd(), file_path, file)
    for i in range(3):
        # Set the start and end times for trimming (in seconds)
        start_time = 30 * i
        end_time = 30 * i + 30
        trimmed_audio = trim_audio(abs_file_path, start_time, end_time)
