## Setup and Environment

Operating System: Ubuntu 24.04
On windows use WSL2.

Run the following command in powershell as an administrator:
```bash
wsl --install -d Ubuntu-24.04
```

Install the following in your Ubuntu environment.

Install python
```bash
sudo apt install python3.12
```

Install `uv`. `uv` is a package manager for python
```bash
curl LsSf https://astral.sh/uv/0.13.0/install.sh | sh
```

Install `dvc`
```bash
uv tool install dvc
```

Create a python virtual environment and activate it.
**Always make sure your venv is activated when you are running code.**
```bash
uv venv
source .venv/bin/activate
```

Install HuggingFace CLI
```bash
uv pip install huggingface_hub
```

Create a new directory and download the dataset. You will need at least 33 GB of free space.
```bash
mkdir data/raw/scipostlayout
hf download omron-sinicx/scipostlayout_v2 --repo-type dataset --local-dir data/raw/scipostlayout
```

Unzip the dataset. You will need another 33 GB of space
```bash
mkdir data/interim
unzip data/raw/scipostlayout/scipostlayout.zip -d data/interim/scipostlayout
```

Install requirements.
```bash
uv pip install -r requirements.txt
```

Register the virtual environment as a jupyter kernel.
```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=my-uv-env   
```

Start jupyter lab in a separate terminal **outside the venv**.
```bash
jupyter lab
```
Go to the url shown in the terminal, or go to http://localhost:8888/lab.
Open a notebook, set the kernel to infographix-venv, and run something to test that it's working.
Also test whether when you use the default python-3 kernel, you get a `ModuleNotFoundError` when you try to import a library (for example, `tqdm`).

**Before every commit, make sure to update requirements.txt**
**Make sure you are in the activated virtual environment before running this:**
```bash
uv pip freeze > requirements.txt
```

**The below commands are not to be run**

Initialize DVC
```bash
dvc init
```
