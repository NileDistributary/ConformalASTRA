@echo off
REM Install gdown if not already installed
pip show gdown >nul 2>&1
if errorlevel 1 (
    echo Installing gdown...
    pip install gdown
)

REM Download Pretrained ASTRA Models
if not exist pretrained_astra_weights mkdir pretrained_astra_weights
cd pretrained_astra_weights
python -m gdown 1k5XclP7XRwiJOXkB7QJUn9OSDuRWEd8c -O pretrained_astra_weights.zip
tar -xf pretrained_astra_weights.zip
del pretrained_astra_weights.zip
cd ..
echo Done! Pretrained ASTRA weights downloaded.