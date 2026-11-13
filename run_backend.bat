@echo off
echo Starting Infographix Backend via WSL...
wsl -d Ubuntu-24.04 bash -lc "cd /home/raksh/projects/infographix && ./run_backend.sh"

