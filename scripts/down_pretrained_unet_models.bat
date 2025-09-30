@echo off
REM Install gdown if not already installed
pip show gdown >nul 2>&1
if errorlevel 1 (
    echo Installing gdown...
    pip install gdown
)

REM Download U-Net Pretrained Embeddings
if not exist pretrained_unet_weights mkdir pretrained_unet_weights
cd pretrained_unet_weights
python -m gdown 1ygi7-XtVn_24MfUxZ1-OrswSsm3z1eQ1 -O pretrained_unet_weights.zip
tar -xf pretrained_unet_weights.zip
del pretrained_unet_weights.zip
cd ..
echo Done! Pretrained U-Net weights downloaded.