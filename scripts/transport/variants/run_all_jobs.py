import os

files = os.listdir()
for file in files:
    if file.endswith('.slurm'):
        os.system(f'sbatch {file}')