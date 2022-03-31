# Multimodal Feature Learning


## Setup

* Change Permissions
```sh
chmod 755 download.sh
chmod -R 1777 /home/arnavshah/multimodal-feature-learning/submodules/pycocoevalcap/tokenizer/
```

* Create container
```sh
docker container run -it --gpus all --name btech-it-dvc -P --network=host -v /home/arnavshah:/home/arnavshah nvcr.io/nvidia/pytorch:22.01-py3
```

* Attach
```sh
docker attach btech-it-dvc
```

* Install Dependencies
```sh
pip install -r requirements.txt
python -m spacy download en
wandb login

apt-get update
apt update
apt-get update
apt install default-jre
```