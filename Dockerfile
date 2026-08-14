FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_opencv

USER root
RUN apt-get update && apt-get install -y curl wget ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Dell CA root certificate bundle (fixes SSL behind corporate proxy)
COPY dellca2018-bundle.crt /usr/local/share/ca-certificates/dellca2018-bundle.crt
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

ENV DL_PYTHON_EXECUTABLE=/usr/bin/python3
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=300
ARG PIP_INDEX_URL=https://artifacts.dell.com/artifactory/api/pypi/python/simple
ARG PIP_TRUSTED_HOST="artifacts.dell.com pypi.org files.pythonhosted.org"
RUN $DL_PYTHON_EXECUTABLE -m pip install \
                'torch==2.0.1' \
                'torchvision==0.15.2' \
                'torchaudio==2.0.2' \
                onnxruntime \
                dtlpy \
                opencv-python \
                pycocotools \
                matplotlib \
                onnx \
                numpy==1.26.4 \
                urllib3>=2.5.0 \
                protobuf>=4.25.8 \
                setuptools>=78.1.1

RUN $DL_PYTHON_EXECUTABLE -m pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'

# make the artifacts available for all users
RUN mkdir -p /tmp/app /tmp/app/artifacts && chmod -R 0777 /tmp/app
RUN wget -O /tmp/app/artifacts/sam2_hiera_small.pt https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_small.pt

# docker build --platform linux/amd64 --no-cache -f Dockerfile -t hub.dataloop.ai/dtlpy-runner-images/sam:0.1.61 .
# docker push hub.dataloop.ai/dtlpy-runner-images/sam:0.1.61
