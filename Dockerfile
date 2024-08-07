FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.pytorch2
USER root
RUN apt update && apt install -y curl

USER 1000

COPY ./ /tmp/app
WORKDIR /tmp/app

RUN pip3 install --user -r _requirements.txt


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/cpu/grounded-sam:0.1.2 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/cpu/grounded-sam:0.1.2
