FROM nvcr.io/nvidia/pytorch:22.03-py3

RUN apt update && \
    apt install -y \
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
  h5py

ARG GID=2040
ENV GROUP="arnavshah"
RUN groupadd -g $GID $GROUP && usermod -aG $GROUP root


CMD ["/usr/bin/zsh"]
