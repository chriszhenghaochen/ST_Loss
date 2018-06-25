import pandas as pd

data = pd.read_csv('path_to_your_file.txt', sep=".", header=None)

data = data[0]

data.to_csv('ouput.txt', sep='\t', index=False, header=False)