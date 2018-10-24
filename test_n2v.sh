#!/bin/sh
#SBATCH --cpus-per-task=12
#SBATCH --job-name=n2v-test
#SBATCH --ntasks=1
#SBATCH --time=0-00:10:00
#SBATCH --mail-user=alexandre.kabbach@unige.ch
#SBATCH --mail-type=ALL
#SBATCH --partition=debug
#SBATCH --clusters=baobab
#SBATCH --output=slurm-%J.out

module load GCC/6.4.0-2.28  OpenMPI/2.1.2 Python/3.6.4 TensorFlow/1.7.0-Python-3.6.4 cuDNN/7.0.5-CUDA-9.1.85

srun python3 install /home/kabbach/nonce2vec/setup.py

srun n2v train --data /home/kabbach/nonce2vec/data/enwiki.20180920.utf8.lower.txt --outputdir /home/kabbach/nonce2vec/models --vocab /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.vocab --alpha 0.025 --neg 5 --window 5 --epochs 1 --size 400 --min-count 50 --sample 1e-5  --train-mode skipgram --t-num-threads 48 --p-num-threads 48 --batch 2097152
