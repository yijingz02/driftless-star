# Container Image Use Overview

## Docker

### Building Docker container images

From the repo root

```
docker build --file Dockerfile --build-arg <build-args> --tag <tag> <context path>
```

Example:

```
docker build \
    --file Dockerfile \
    --build-arg ENVIRONMENT="stage-1-vmec" \
    --tag ghcr.io/rkhashmani/stellaforge:stage-1-vmec-cpu \
    .
```

### Pulling Docker container images from a registry

```
docker pull <registry>/<repository>/<image>:<tag>
```

Example:

```
docker pull ghcr.io/rkhashmani/stellaforge:stage-1-vmec-cpu
```

### Running Docker container images

```
docker run --rm -ti [--volume <local mount path>:<container-side mount path>] [--gpus <gpu>] <container>
```

Examples:

* Run a `stage-1-vmec-cpu` container in an interactive shell

```console
$ docker run --rm -ti ghcr.io/rkhashmani/stellaforge:stage-1-vmec-cpu bash
root@060500d71aaf:/app# command -v python
/app/.pixi/envs/stage-1-vmec/bin/python
```

* Run a `stage-1-vmec-cpu` container in an interactive shell with the local working directory mounted

```console
$ docker run --rm -ti -v $PWD:/work -w /work ghcr.io/rkhashmani/stellaforge:stage-1-vmec-cpu bash
root@1d603f18cd72:/work# pwd
/work
root@1d603f18cd72:/work# command -v python
/app/.pixi/envs/stage-1-vmec/bin/python
```

* Execute a command in a `stage-1-vmec-cpu` container

```console
$ docker run --rm -ti ghcr.io/rkhashmani/stellaforge:stage-1-vmec-cpu python -c 'import vmec_jax; print(vmec_jax)'
<module 'vmec_jax' from '/app/.pixi/envs/stage-1-vmec/lib/python3.14/site-packages/vmec_jax/__init__.py'>
```

* Run a `stage-1-vmec-gpu` container in an interactive shell with [NVIDIA driver support](https://github.com/NVIDIA/nvidia-container-toolkit)

```console
$ docker run --rm -ti --gpus all ghcr.io/rkhashmani/stellaforge:stage-1-vmec-gpu bash
root@24982ece960b:/app# nvidia-smi --version
NVIDIA-SMI version  : 590.48.01
NVML version        : 590.48
DRIVER version      : 590.48.01
CUDA Version        : 13.1
```

## Apptainer

On Linux machines Apptainer can be installed from conda-forge with

```
pixi global install apptainer
```

### Building Apptainer container images

Apptainer has no concept of "context" and so requires you to operate from a the directory of the Apptainer definition file expects to be executed from.

```
apptainer build [local options...] <IMAGE PATH> <BUILD SPEC>
```

Example:

```
apptainer build \
    --build-arg ENVIRONMENT="stage-1-vmec" \
    stage-1-vmec.sif \
    apptainer.def
```

### Pulling Apptainer container images from a registry

```
apptainer pull [pull options...] [output file] <URI>
```

where for OCI container registries, the `<URI>` follows

```
oras://<registry>/<repository>/<image>:<tag>
```

Example:

```
apptainer pull stage-1-vmec-cpu.sif oras://ghcr.io/rkhashmani/stellaforge:apptainer-stage-1-vmec-cpu
```

### Running Apptainer container images

```
apptainer run [run options...] <container> [args...]
```

Examples:

* Run a `stage-1-vmec-cpu` container in an interactive shell

```console
$ apptainer run --containall --writable-tmpfs ./stage-1-vmec-cpu.sif
(mvp:stage-1-vmec)
```

* Run a `stage-1-vmec-cpu` container in an interactive shell with the local working directory mounted

```console
$ apptainer run --containall --writable-tmpfs --bind "$PWD":/work --pwd /work ./stage-1-vmec-cpu.sif
(mvp:stage-1-vmec) pwd
/work
(mvp:stage-1-vmec)
```

* Execute a command in a `stage-1-vmec-cpu` container

```console
$ apptainer run --containall --writable-tmpfs ./stage-1-vmec-cpu.sif python -c 'import vmec_jax; print(vmec_jax)'
<module 'vmec_jax' from '/app/.pixi/envs/stage-1-vmec/lib/python3.14/site-packages/vmec_jax/__init__.py'>
```

* Run a `stage-1-vmec-gpu` container in an interactive shell with [NVIDIA driver support](https://apptainer.org/docs/user/latest/gpu.html)

```console
$ apptainer run --containall --writable-tmpfs --nv ./stage-1-vmec-gpu.sif
(mvp:stage-1-vmec-gpu) nvidia-smi --version
NVIDIA-SMI version  : 590.48.01
NVML version        : 590.48
DRIVER version      : 590.48.01
CUDA Version        : 13.1
```
