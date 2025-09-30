# Activate your virtual environment if needed
# .\venv310\Scripts\activate

# Step 1: Create dataset directories
New-Item -ItemType Directory -Force -Path datasets\eth_ucy
New-Item -ItemType Directory -Force -Path datasets\eth_ucy\homography

# Step 2: Download ETH dataset zip file using gdown
python -m gdown 1qcWaVpyeosiirNP7pVyxHmeiJfzwnkKr -O eth_dataset.zip

# Step 3: Extract the dataset
Expand-Archive -Path "eth_dataset.zip" -DestinationPath datasets\eth_ucy

# Step 4: Download homography files using gdown
python -m gdown 15hDOhjSJbKtV9ys1D_9tGFyKpt6Bv2eJ -O datasets/eth_ucy/homography/H_eth.txt
python -m gdown 1LwQCCpjBGaHpIQBNU1b5iA7d8CfxzhoE -O datasets/eth_ucy/homography/H_hotel.txt
python -m gdown 1Mnb9sJB7TmDHWYDgYxN-lvIQSw9p4gJH -O datasets/eth_ucy/homography/H_students003.txt
python -m gdown 1FYGpoFAoIzmkpSjv5jSJfsfjWfnBlmqK -O datasets/eth_ucy/homography/H_zara01.txt
python -m gdown 1x9VWY2H3IV-6MT60TKOFb0A5Tp8yaSMP -O datasets/eth_ucy/homography/H_zara02.txt

# Step 5: Convert videos to image frames
python utils/video2images.py

# Step 6: Preprocess ETH/UCY annotations
foreach ($subset in @("eth", "hotel", "univ", "zara01", "zara02")) {
    python data/process_eth.py --subset $subset
}

# Step 7: Clean up
Remove-Item -Force "eth_dataset.zip"

Write-Host ""
Write-Host "ETH dataset setup completed successfully." -ForegroundColor Green
