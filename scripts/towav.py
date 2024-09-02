import os
import polars as pl
from midi2audio import FluidSynth
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

fs = FluidSynth("C:\\ProgramData\\soundfonts\\default.sf2")
df = pl.read_csv("..\\filtered.csv")

def midi_to_wav_path(midi_path):
    directory, filename = os.path.split(midi_path)
    new_directory = directory.replace('midis', 'wavs', 1)
    new_filename = os.path.splitext(filename)[0] + '.wav'
    wav_path = os.path.join(new_directory, new_filename)
    return wav_path

midis = df["file_paths"].to_list()
wavs = list(map(midi_to_wav_path, midis))

for midi, wav in zip(midis, wavs):
    fs.midi_to_audio(midi, wav)
    audio = AudioSegment.from_wav(wav)
    
    # Detect non-silent parts (returns list of (start, end) tuples in milliseconds)
    non_silent_ranges = detect_nonsilent(audio, min_silence_len=1000, silence_thresh=audio.dBFS-16)
    
    if non_silent_ranges:
        # If non-silent part detected, trim the silence from the start
        start_trim = non_silent_ranges[0][0]
        trimmed_audio = audio[start_trim:]
    else:
        trimmed_audio = audio
    
    trimmed_audio = trimmed_audio[:30000]
    trimmed_audio.export(wav, format="wav")