FROM dataloopai/dtlpy-agent:cpu.py3.10.pytorch2

USER root
RUN apt update && apt install -y curl

USER 1000

COPY ./ /tmp/app
WORKDIR /tmp/app

RUN pip3 install --user -r _requirements.txt
RUN pip3 install git+https://github.com/facebookresearch/segment-anything-2.git --no-build-isolation

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/grounded-sam:0.1.3 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/grounded-sam:0.1.3
