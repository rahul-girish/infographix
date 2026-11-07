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

Install DVC
```bash
uv pip install dvc dvc-gdrive
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
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=infographix-venv  
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

Note that all DVC commands can only be run from within the virtual environment.
Initialize DVC
```bash
dvc init
```

In Google Drive, create a folder. Copy the folder ID (The part of the URL that comes after `folder/`). Then,
```bash
dvc remote add --default gdrive gdrive://folder_id
```
Check if it was configured:
```bash
dvc remote list
```

Now, I faced some issues with dependencies. There was a version mismatch between PyOpenSSL and cryptography, and I spent a painfully long time trying to fix it. In the end, I had to remove my installation of DVC and install afresh using pip.

Then there came an authentication error from Google Auth. When using Google Drive as a remote, DVC requests the `drive.files` OAuth scope, which Google changed their policy about and is now considered sensitive. As a result, the process of using Google Drive became much more lengthy.

The following paragraphs summarize that process:
Go to Google Cloud Console, create a new project. Name it whatever. Under APIs and Services, go to Library and enable Google Drive API. Also, I had to create a new OAuth client, following which I saved the client ID and secret. I also had to add the emails of the team members to the list of test users. After all this, the following needs to be configured in DVC.

```bash
dvc remote modify gdrive --local gdrive_client_id 'client_id'
dvc remote modify gdrive --local gdrive_client_secret 'client_secret'
```


