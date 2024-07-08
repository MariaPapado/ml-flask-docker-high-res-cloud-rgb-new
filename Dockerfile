from python:3.8-slim
WORKDIR /api
COPY requirements.txt .


#RUN yes | apt-get clean && apt-get update
#RUN yes | apt-get install libgeos-dev libpq-dev
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN yes | ln -s /usr/bin/python3 /usr/bin/python && python -m pip install -r requirements.txt
# CUDA
#RUN yes | pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 --default-timeout=3600
RUN yes | pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 segmentation_models_pytorch pytorch_lightning --extra-index-url https://download.pytorch.org/whl/cpu
COPY app.py .
CMD python app.py
