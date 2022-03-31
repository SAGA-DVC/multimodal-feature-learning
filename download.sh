#!/bin/sh
pip install -r requirements.txt
python -m spacy download en
wandb login

# apt update
# apt-get update
# apt install default-jre

# chmod 755 ./download.sh
# chmod -R 1777 /home/arnavshah/multimodal-feature-learning/submodules/pycocoevalcap/tokenizer/