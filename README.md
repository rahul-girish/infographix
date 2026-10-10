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

**The below commands are not to be run**

Initialize DVC
```bash
dvc init
```
