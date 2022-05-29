FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt update && \
    apt-get update && \
	apt-get install --fix-missing && \
    apt install -y \
    default-jre \
    htop \
    zsh \
    && \
    rm -rf /var/lib/apt/lists/*

RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

RUN git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && \
    ~/.fzf/install --all

RUN pip install ml-collections \
  av \
  timm \
  wandb \
  h5py \
  pytorchvideo \
  spacy

RUN python -m spacy download en

CMD ["/usr/bin/zsh"]
