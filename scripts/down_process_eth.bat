@echo off
REM Install gdown if not already installed
pip show gdown >nul 2>&1
if errorlevel 1 (
    echo Installing gdown...
    pip install gdown
)

REM Download ETH videos and annotations
echo Downloading ETH dataset...
python -m gdown 1qcWaVpyeosiirNP7pVyxHmeiJfzwnkKr

REM Unzip
if not exist datasets\eth_ucy mkdir datasets\eth_ucy
echo Extracting ETH dataset...
tar -xf eth_dataset.zip -C datasets\eth_ucy

REM Download homography matrices
if not exist datasets\eth_ucy\homography mkdir datasets\eth_ucy\homography
echo Downloading homography matrices...
python -m gdown 1omNDA7T6mGGF7PCUNOae_OaK_g-DTYMH -O datasets\eth_ucy\homography\H_eth.txt
python -m gdown 1uZkQ1_jEaJx1YESlXJjAwLhY-IwVKr5H -O datasets\eth_ucy\homography\H_hotel.txt
python -m gdown 1q-TThCER2UJFZzoShpJxKkDgfrf9xgTX -O datasets\eth_ucy\homography\H_students003.txt
python -m gdown 1O-BL8RdqS4LwC8aiDA37adH_6w7JAX-Q -O datasets\eth_ucy\homography\H_zara01.txt
python -m gdown 1CAhcHdnjyozMmV8THhulymBRefx9SN5D -O datasets\eth_ucy\homography\H_zara02.txt

REM Convert videos to images
echo Converting videos to images...
python utils/video2images.py

REM Process annotations for each dataset
echo Processing annotations...
for %%d in (eth hotel univ zara01 zara02) do (
    echo Processing %%d...
    python data/process_eth.py --subset %%d
)

REM Remove zip file
del eth_dataset.zip
echo Done! ETH dataset downloaded and processed.