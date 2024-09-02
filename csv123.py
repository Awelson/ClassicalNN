import os
import polars as pl

directory = "midis"

def extract_composer(filename):
    return filename.split(',')[0].strip() + ", " + filename.split(',')[1].strip()

file_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
composers = list(map(extract_composer, os.listdir(directory)))

data = {
    "file_paths": file_paths,
    "composers": composers
}

df = pl.DataFrame(data)
df.write_csv("mididata.csv")