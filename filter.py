import polars as pl
from pretty_midi import PrettyMIDI

def duration(file_path) : 
    midi = PrettyMIDI(file_path)
    return midi.get_end_time()

df = pl.read_csv("mididata.csv")

composer_counts = df.group_by("composers").agg(
    pl.col("file_paths").count().alias("counts")
)

print(composer_counts.sort(by="counts", descending=True).head(5))

composers_of_interest = ['Liszt, Franz', 'Scarlatti, Domenico', 'Bach, Johann Sebastian', 'Schubert, Franz', 'Chopin, Frederic']

result_df = (
    df
    .filter(pl.col('composers').is_in(composers_of_interest))
    .with_columns(pl.col('file_paths').map_elements(duration).alias("durations"))
    .filter(pl.col('durations')>=30)
    .sort(by=["composers", "durations"])
    .group_by('composers').head(80)
)

result_df.write_csv('filtered.csv')