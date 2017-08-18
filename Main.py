import os

data_dir = "Data"
 
for directories, subdirs, files in os.walk(data_dir):
    print(directories, subdirs, len(files))