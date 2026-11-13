# Run Infographix Frontend inside WSL from PowerShell
Write-Host "⚡ Starting Infographix Frontend via WSL..." -ForegroundColor Green
wsl -d Ubuntu-24.04 bash -lc "cd /home/raksh/projects/infographix/frontend && npm run dev"

