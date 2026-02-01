FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /workspace
COPY /src/ /workspace/

RUN apt update && apt install -y git ffmpeg libsm6 libxext6

RUN git lfs install
RUN git clone https://huggingface.co/deepseek-ai/DeepSeek-OCR

RUN python -m pip install --upgrade pip 
RUN pip install -r requirements.txt
RUN pip install flash-attn==2.7.3 --no-build-isolation

WORKDIR /workspace
CMD ["bash"]
