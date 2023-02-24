import logging
import os
from pydub import AudioSegment
import librosa
import soundfile as sf
# ======================================
inputs_folder = "inputs"
outputs_folder = "outputs"
allowed_formats = ["mp3", "flac"]


def mp3_to_wav_all_files(inputs_folder=inputs_folder, outputs_folder=outputs_folder):
    # inputs_folder_path = os.path.join(os.getcwd())  # , inputs_folder
    # input_file_path = os.path.join(os.getcwd(), inputs_folder, file_path_music)
    for file in os.listdir(os.getcwd()):
        if file.endswith('.mp3'):
            audio_file = inputs_folder_path + "\\" + file.title()
            print("Processing {0}".format(file.title()))
            print("\n =========== \n")
            audio, sr = librosa.load(audio_file)
            sf.write("outputs/{0}.wav".format(file), audio, sr)


def audio_to_wav(file_path, outputs_folder=outputs_folder):
    try:
        file_format = file_path.split('.')[-1]
        assert file_format in allowed_formats
        file_name = file_path.split('\\')[-1]
        print("\n =========== ")
        print("Processing {0}".format(file_name))
        print(" =========== \n")
        audio, sr = sf.read(file_path)
        output_file = "{0}.wav".format(file_name)
        sf.write("outputs/{0}".format(output_file), audio, sr)
        print("Conversion done ! ")
        print(output_file)

    except AssertionError:
        print('DUT DUT ERROR, THIS IS NOT A MP3 FILE')
        exit(1)

    return os.path.join(os.getcwd(), "outputs", output_file)


# Zone test
audio_file = os.path.join(os.getcwd(), inputs_folder,
                          "sample3.flac")
outputs_folder = "outputs"
new_file_path = audio_to_wav(audio_file, outputs_folder)
print(new_file_path)
