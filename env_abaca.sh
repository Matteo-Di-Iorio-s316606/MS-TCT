#!/bin/bash

ENV_NAME="mstct"

# Inizializza conda
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "ERRORE: Conda non trovata."
    exit 1
fi

# Attiva environment
conda activate $ENV_NAME

echo "Environment $ENV_NAME attivato."
echo "Python: $(which python)"
echo "Torch CUDA disponibile: $(python -c 'import torch; print(torch.cuda.is_available())')"