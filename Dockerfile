FROM dataloopai/dtlpy-agent:cpu.py3.8.pytorch2
USER root
RUN apt update
RUN apt install -y git-lfs

USER 1000

COPY ./ /tmp/app
WORKDIR /tmp/app

RUN pip3 install --user -r _requirements.txt


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/grounded-sam:0.1.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/grounded-sam:0.1.1
