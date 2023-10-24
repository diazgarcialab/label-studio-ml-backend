FROM python:3.11-slim

WORKDIR /app

RUN apt-get -y update \
    && apt-get -y install git wget ffmpeg libsm6 libxext6 libffi-dev python3-dev gcc

ENV PYTHONUNBUFFERED=True \
    PORT=9090 \
    WORKERS=2 \
    THREADS=4

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN  git clone https://github.com/IDEA-Research/GroundingDINO.git \
    && cd GroundingDINO \
    && pip -q install -e . \
    && mkdir weights \
    && cd weights \
    && wget -q  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

RUN wget -q https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt \
   && wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

COPY . ./

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi:app