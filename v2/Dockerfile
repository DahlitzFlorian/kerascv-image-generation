FROM almalinux:8

ARG PYTHON_VERSION=3.9
ARG CUDA_VERSION=11.8.0

USER root

# set prefix
ENV CONDA_DIR=/opt/conda
ENV MAMBA_ROOT_PREFIX=${CONDA_DIR}
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# create conda directory
RUN mkdir -p "${CONDA_DIR}"

RUN dnf -y groupinstall "Development Tools"

# openblas (?)
RUN dnf config-manager --enable powertools

RUN dnf install \
    -y \
    bzip2 \
    openblas-devel \
    wget

WORKDIR /root

# Install micromamba
RUN wget -qO /tmp/micromamba.tar.bz2 \
        "https://micromamba.snakepit.net/api/micromamba/linux-ppc64le/latest" && \
    tar -xvjf /tmp/micromamba.tar.bz2 --strip-components=1 bin/micromamba && \
    rm /tmp/micromamba.tar.bz2

# Install CUDA
RUN ./micromamba install \
    --yes \
    --root-prefix="${CONDA_DIR}" \
    --prefix="${CONDA_DIR}" \
    -c nvidia \
    cuda

# Install dependencies
RUN ./micromamba install \
    --yes \
    --root-prefix="${CONDA_DIR}" \
    --prefix="${CONDA_DIR}" \
    -c rocketce \
    -c defaults \
    -c anaconda \
    python=${PYTHON_VERSION} \
    cmake \
    pillow \
    pip \
    pytorch-base

RUN pip install --upgrade pip

RUN pip install --prefer-binary \
    diffusers \
    transformers \
    accelerate \
    safetensors

COPY t.py t.py

CMD ["/bin/bash"]
