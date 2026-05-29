<img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/nnInteractive_header.png"  width="1200">

# nnInteractive: Redefining 3D Promptable Segmentation

This repository contains the napari plugin for nnInteractive. Check out the
[python backend](https://github.com/MIC-DKFZ/nnInteractive) and [MITK integration](https://www.mitk.org/MITK-nnInteractive) for more.

## What is nnInteractive?

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025). nnInteractive: Redefining 3D Promptable Segmentation. https://arxiv.org/abs/2503.08373 \
> \*: equal contribution

Link: [![arXiv](https://img.shields.io/badge/arXiv-2503.08373-b31b1b.svg)](https://arxiv.org/abs/2503.08373)

##### Abstract:

Accurate and efficient 3D segmentation is essential for both clinical and research applications.
While foundation models like SAM have revolutionized interactive segmentation, their 2D design and domain shift limitations make them ill-suited for 3D medical images.
Current adaptations address some of these challenges but remain limited, either lacking volumetric awareness, offering restricted interactivity, or supporting only a small set of structures and modalities.
Usability also remains a challenge, as current tools are rarely integrated into established imaging platforms and often rely on cumbersome web-based interfaces with restricted functionality.
We introduce nnInteractive, the first comprehensive 3D interactive open-set segmentation method.
It supports diverse prompts—including points, scribbles, boxes, and a novel lasso prompt—while leveraging intuitive 2D interactions to generate full 3D segmentations.
Trained on 120+ diverse volumetric 3D datasets (CT, MRI, PET, 3D Microscopy, etc.), nnInteractive sets a new state-of-the-art in accuracy, adaptability, and usability.
Crucially, it is the first method integrated into widely used image viewers (e.g., Napari, MITK), ensuring broad accessibility for real-world clinical and research applications.
Extensive benchmarking demonstrates that nnInteractive far surpasses existing methods, setting a new standard for AI-driven interactive 3D segmentation.

<img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/figure1_method.png" width="1200">

## Demo Videos

<a href="https://www.youtube.com/watch?v=H_L6LL0FRoo">
    <img src="https://img.youtube.com/vi/H_L6LL0FRoo/0.jpg" width="270">
</a>
<a href="https://www.youtube.com/watch?v=YoMZ7Xv7gKI">
    <img src="https://img.youtube.com/vi/YoMZ7Xv7gKI/0.jpg" width="270">
</a>
<a href="https://www.youtube.com/watch?v=V0rqPYA3sjA">
    <img src="https://img.youtube.com/vi/V0rqPYA3sjA/0.jpg" width="270">
</a>

## Installation

### Prerequisites

For **local** inference you need a Linux or Windows computer with an Nvidia GPU. 10GB of VRAM is recommended. Small objects should work with \<6GB.

> [!TIP]
> **On a Mac, or without a capable Nvidia GPU?** Use [**Remote inference**](#remote-inference-server--client) instead. nnInteractive relies heavily on 3D convolutions, which are prohibitively slow on Apple Silicon (MPS) and CPU hardware — running the model on a remote GPU and driving it from napari is the recommended, and often only practical, way to use nnInteractive on these machines. The napari client itself has no GPU requirements and runs anywhere; only the GPU server needs the hardware above.

##### 1. Create a virtual environment:

nnInteractive supports Python 3.10+ and works with Conda, pip, or any other virtual environment. Here’s an example using Conda:

```
conda create -n nnInteractive python=3.12
conda activate nnInteractive
```

##### 2. Install the correct PyTorch for your system

Go to the [PyTorch homepage](https://pytorch.org/get-started/locally/) and pick the right configuration.
Note that since recently PyTorch needs to be installed via pip. This is fine to do within your conda environment.

For Ubuntu with a Nvidia GPU, pick 'stable', 'Linux', 'Pip', 'Python', 'CUDA12.6' (if all drivers are up to date, otherwise use and older version):

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

##### 3. Install this repository + dependencies via

Install napari if necessary

```bash
pip install napari[all]
```

Install the plugin via pip:

```bash
pip install napari-nninteractive
```

Or clone and install this repository:

```bash
git clone https://github.com/MIC-DKFZ/napari-nninteractive
cd napari-nninteractive
pip install -e .
```

**Note:** Model weights are automatically downloaded on first use. This can take up to a couple of minutes depending on your internet connection

## Getting Started

Use one of these three options to start napari and activate the plugin.
Afterward, Drag and drop your images into napari.

\***Note if getting asked which plugin to use for opening .nii.gz files use napari-nifti.**

a) Start napari, then Plugins -> nnInteractive.

```
napari
```

b) Run this to start napari with the plugin already started.

```
napari -w napari-nninteractive
```

c) Run this to start napari with the plugin and open an image directly

```
napari demo_data/liver_145_0000.nii.gz -w napari-nninteractive
```

# How to use

**Note:** To open Nifti (.nii.gz, .nii) files we recommend to select napari-nifti.

<img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/gui_instuctions.png" width="1200">

## Remote inference (server / client)

**This is the recommended way to run nnInteractive on a Mac or any machine without a capable Nvidia GPU.** nnInteractive
depends heavily on 3D convolutions, which are prohibitively slow on Apple Silicon (MPS) and CPU — in practice, remote
inference is the only usable option on those systems.

You run the model on a remote machine and drive it from napari over the network. One server hosts the model and serves
multiple napari clients; each client gets its own independent session (image, prompts, segmentation), while the
model weights are loaded once and shared. The napari client has no GPU requirements, so it runs fine on a laptop or Mac.

### On the GPU machine — start the server

```bash
nninteractive-server \
    --model-dir /path/to/checkpoint_folder \
    --fold 0 \
    --host 0.0.0.0 --port 1527 \
    --max-sessions 4 \
    --api-key "$(openssl rand -hex 32)"
```

Share the printed API key with your users via whatever channel you use for shared credentials. For a single-user setup over SSH (no API key, no exposed port), see the SSH-tunnel section in the full docs linked below.

### In napari — connect to the server

1. Start napari with the plugin (`napari -w napari-nninteractive`).
2. In the plugin panel, flip the **Local | Remote** switch to **Remote**.
3. Enter the **Server URL** (e.g. `http://gpu-box.lab:1527`) and the **API key**.
4. Click **Connect**. On success the status line shows `connected (...)`.
5. Use **Initialize** and the rest of the workflow exactly as in local mode — point, bbox, scribble, and lasso prompts all behave the same; only the prediction runs on the remote GPU.

The model checkpoint is fixed by the server at startup, so the local checkpoint selector is hidden in Remote mode.

### Things to be aware of

- **Lost connection or idle timeout.** Server-side sessions are reaped after 10 minutes of inactivity (configurable on the server), and a server that goes away (restart, crash, network drop) ends the session too. When this happens the plugin resets the **Connect** button and asks you to reconnect. **Your current segmentation is preserved** on the client: after you reconnect and click **Initialize** again, the image is re-uploaded and the segmentation is restored so you can keep refining where you left off. Only the in-progress prompt markers (the individual points/boxes/scribbles) need to be redone.
- **Server full.** If all session slots are taken, Connect reports `server is full, try again later`. Retry shortly.
- **Concurrent users.** Multiple researchers can use the same server independently. Predictions are serialized on the GPU, so heavy concurrent use makes individual predictions wait briefly — scale out by running one server per GPU.
- **Corporate proxies.** If your client machine sets `HTTP_PROXY` / `HTTPS_PROXY`, add the server host to `NO_PROXY`, otherwise requests get intercepted and Connect will report an HTML/proxy error.

For full details — all server flags, authentication, TLS / reverse-proxy setup, SSH tunnels, troubleshooting — see [SERVER_CLIENT.md in the nnInteractive backend repo](https://github.com/MIC-DKFZ/nnInteractive/blob/master/SERVER_CLIENT.md).

## Citation

When using nnInteractive, please cite the following paper:

> Isensee, F.\*, Rokuss, M.\*, Krämer, L.\*, Dinkelacker, S., Ravindran, A., Stritzke, F., Hamm, B., Wald, T., Langenberg, M., Ulrich, C., Deissler, J., Floca, R., & Maier-Hein, K. (2025). nnInteractive: Redefining 3D Promptable Segmentation. https://arxiv.org/abs/2503.08373 \
> \*: equal contribution

Link: [![arXiv](https://img.shields.io/badge/arXiv-2503.08373-b31b1b.svg)](https://arxiv.org/abs/2503.08373)

# License

Note that while this repository is available under Apache-2.0 license (see [LICENSE](./LICENSE)), the [model checkpoint](https://huggingface.co/nnInteractive/nnInteractive) is `Creative Commons Attribution Non Commercial Share Alike 4.0`!

______________________________________________________________________

## Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/napari-nninteractive/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

[copier]: https://copier.readthedocs.io/en/stable/
[napari]: https://github.com/napari/napari
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
