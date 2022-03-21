# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:22.03-py3
RUN pip install -r requirements.txt
CMD ["bash"]