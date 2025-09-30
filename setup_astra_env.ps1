# Activate your virtual environment first if needed
# .\venv310\Scripts\activate

Write-Host "Installing basic requirements..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Installing PyTorch Geometric extensions..." -ForegroundColor Cyan
$wheelURL = "https://data.pyg.org/whl/torch-2.0.0+cu117.html"
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f $wheelURL

Write-Host "Skipping pyg_lib (not available for Windows)..." -ForegroundColor Yellow

Write-Host "Installing missing extras..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install munch pyyaml

Write-Host ""
Write-Host "ASTRA environment setup complete." -ForegroundColor Green
