# Multi GPU training guide

1. Connect to a GPU instance (locally OR using a runpod)
2. Set up a virtual environment and run the installs:

```
python -m venv multigpuvenv
source multigpuvenv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
export HF_HUB_ENABLE_HF_TRANSFER=True
huggingface-cli login 
```

> flash-attn has to be installed separately (as per the commands above) or can give issues if installed from the requirements file.

# How to use Runpod

SSH into the pod. To do this you'll need:

1. to generate a public-private key pair in the .ssh folder on your local device.
2. copy and paste the public key into Runpod Settings.
3. When the pod has started, copy the TCP connection details into your VSCode ssh configuration to then connect to the GPU
