FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_opencv

USER root
RUN apt update && apt install -y curl

ENV DL_PYTHON_EXECUTABLE=/usr/bin/python3
ENV PIP_NO_CACHE_DIR=1
RUN $DL_PYTHON_EXECUTABLE -m pip install --upgrade pip
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
                # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
                urllib3>=2.5.0 \
                protobuf>=4.25.8 \
                setuptools>=78.1.1 \
                einops \
                decord

RUN $DL_PYTHON_EXECUTABLE -m pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'
RUN $DL_PYTHON_EXECUTABLE -m pip install 'git+https://github.com/facebookresearch/sam3.git'
# make the artifacts available for all users
RUN mkdir -p /tmp/app /tmp/app/artifacts && chmod -R 0777 /tmp/app
RUN wget -O /tmp/app/artifacts/sam2_hiera_small.pt https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_small.pt


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/sam3:1.0.1 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/apps/sam3:1.0.1
