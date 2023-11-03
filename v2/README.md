# DiffusionPipeline with stabilityai/stable-diffusion-xl-base-1.0

## Setup

1. Install NVIDIA-Driver on host system
1. Install the NVIDIA Container Toolkit ([Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html))

Build the image: 

```shell
podman image build -t t2i .
```

Run the container and pass the GPUs as devices, so the container has access to it:

```shell
podman container run --name t2i -it --device nvidia.com/gpu=all -v huggingface_cache:/root/.cache/huggingface/hub t2i
```
