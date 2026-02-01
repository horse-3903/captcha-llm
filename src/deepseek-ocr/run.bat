docker build -t deepseek-ocr:latest src/deepseek-ocr
docker run -it --gpus all deepseek-ocr bash