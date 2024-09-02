import numpy as np
import librosa.display, os
import matplotlib.pyplot as plt

def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)
    
def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)

create_pngs_from_wavs('../wavs/Schubert', '../spectrograms/Schubert')
create_pngs_from_wavs('../wavs/Liszt', '../spectrograms/Liszt')
create_pngs_from_wavs('../wavs/Bach', '../spectrograms/Bach')
create_pngs_from_wavs('../wavs/Chopin', '../spectrograms/Chopin')
create_pngs_from_wavs('../wavs/Scarlatti', '../spectrograms/Scarlatti')