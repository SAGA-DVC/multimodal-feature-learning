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

<!-- In Submodules -->
## EVAL

* submodules/pycocoevalcap/bleu/bleu.py
```py
score, scores = bleu_scorer.compute_score(option='closest', verbose=0) # verbose=0
```

* submodules/pycocoevalcap/tokenizer/ptbtokenizer.py
```py
p_tokenizer = subprocess.Popen(cmd, cwd=path_to_jar_dirname, \
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) # stderr=subprocess.DEVNULL
```

* evaluation/evaluate.py
```py
class ANETcaptions(object):
    ....

    def evaluate(self):
        ....

        else:
            ....
            # REMOVE if self.verbose
            self.scores['Recall'] = []
            self.scores['Precision'] = []
            for tiou in self.tious:
                precision, recall = self.evaluate_detection(tiou)
                self.scores['Recall'].append(recall)
                self.scores['Precision'].append(precision)
```