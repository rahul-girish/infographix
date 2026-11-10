# Run Infographix Backend inside WSL from PowerShell
Write-Host "🚀 Starting Infographix Backend via WSL..." -ForegroundColor Cyan
wsl -d Ubuntu-24.04 bash -lc "cd /home/raksh/projects/infographix && ./run_backend.sh"

