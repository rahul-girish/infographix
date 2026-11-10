@echo off
echo Starting Infographix Frontend via WSL...
wsl -d Ubuntu-24.04 bash -lc "cd /home/raksh/projects/infographix/frontend && npm run dev"

