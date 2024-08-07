FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.opencv

USER root
RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
RUN python3 -m pip install --upgrade pip
RUN pip install --user 'torch==2.0.1' 'torchvision==0.15.2' 'torchaudio==2.0.2'



COPY ./ /tmp/app
WORKDIR /tmp/app

RUN pip3 install --user -r _requirements.txt
RUN pip3 install git+https://github.com/facebookresearch/segment-anything-2.git --no-build-isolation

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/grounded-sam-adapter:0.1.3 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/grounded-sam-adapter:0.1.3
