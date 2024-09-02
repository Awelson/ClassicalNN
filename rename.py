import os
from unidecode import unidecode

directory = "midis"

name = os.listdir(directory)
newname = list(map(unidecode, name))

for n, nn in zip(name, newname):
    old = os.path.join(directory, n)
    new = os.path.join(directory, nn)
    os.rename(old, new)