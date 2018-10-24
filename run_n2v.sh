#!/bin/sh
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=2
#SBATCH --job-name=n2v-train
#SBATCH --ntasks=1
#SBATCH --time=1-59:59:59
#SBATCH --mail-user=alexandre.kabbach@unige.ch
#SBATCH --mail-type=ALL
#SBATCH --partition=dpnc-gpu
#SBATCH --gres=gpu:titan:6
#SBATCH --clusters=baobab
#SBATCH -e n2v-train-error.e%j
#SBATCH -o n2v-train-out.o%j

module load GCC/7.3.0-2.30  OpenMPI/3.1.1 Python/3.6.6 TensorFlow/1.7.0-Python-3.6.4 cuDNN/7.0.5-CUDA-9.1.85


srun n2v train --data /home/kabbach/nonce2vec/data/enwiki.20180920.utf8.lower.txt --outputdir /home/kabbach/nonce2vec/models --vocab /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.vocab --alpha 0.025 --neg 5 --window 5 --epochs 1 --size 400 --min-count 50 --sample 1e-5  --train-mode skipgram --t-num-threads 48 --p-num-threads 48 --batch 2097152
