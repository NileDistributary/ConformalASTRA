import argparse
import yaml
import munch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from utils.logger import get_logger, create_exp_name
from utils.misc import set_seed, set_device
from data.eth import ETH_dataset
from models.astra_model import ASTRA_model
from models.keypoint_model import UNETEmbeddingExtractor
from models.vae import ConditionalVariationalEncoder
from utils.metrics import AverageMeter

logger = get_logger(__name__)

def train_cvae(cfg):
    # Set device and seed
    device, device_list = set_device(cfg.TRAIN.DEVICE, 1)  # Force batch size 1
    cfg.update(device=device)
    cfg.update(device_list=device_list)
    set_seed(cfg.TRAIN.SEED)

    # Data transforms
    transforms = A.Compose([
        A.LongestMaxSize(cfg.DATA.MIN_RESHAPE_SIZE),
        A.PadIfNeeded(cfg.DATA.MIN_RESHAPE_SIZE, cfg.DATA.MIN_RESHAPE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(cfg.DATA.MEAN, cfg.DATA.STD, max_pixel_value=255.0),
        ToTensorV2()
    ], keypoint_params=A.KeypointParams(format='yx'))

    # Load dataset with batch_size=1
    train_dataset = ETH_dataset(cfg, mode='training', img_transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=1,
                              num_workers=cfg.TRAIN.NUM_WORKERS,
                              shuffle=True, pin_memory=True, drop_last=False)
    logger.info(f"Loaded training dataset with {len(train_dataset)} samples.")

    # Load pretrained ASTRA model
    astra_model = ASTRA_model(cfg)
    astra_model.load_state_dict(torch.load(f'./pretrained_astra_weights/{cfg.SUBSET}_best_model.pth'))
    astra_model = astra_model.to(device)
    astra_model.eval()
    logger.info("Loaded pretrained ASTRA model.")

    # Load pretrained UNET model
    if cfg.MODEL.USE_PRETRAINED_UNET:
        embedding_extractor = UNETEmbeddingExtractor(cfg)
        embedding_extractor.load_state_dict(torch.load('./pretrained_unet_weights/eth_unet_model_best.pt'))
        for param in embedding_extractor.parameters():
            param.requires_grad = False
        embedding_extractor.eval()
        embedding_extractor = embedding_extractor.to(device)
        logger.info("Loaded pretrained U-Net embedding extractor.")
    else:
        embedding_extractor = None

    # Initialize CVAE model
    cvae_model = ConditionalVariationalEncoder(cfg).to(device)
    optimizer = optim.Adam(cvae_model.parameters(), lr=float(cfg.TRAIN.LR))
    criterion = nn.MSELoss()

    # Training loop
    epoch_loss = AverageMeter()
    for epoch in range(int(cfg.TRAIN.NUM_EPOCH)):
        cvae_model.train()
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        for batch_idx, batch in loop:
            past_loc, fut_loc, num_valid, imgs, gt_maps, traj_coords = batch

            # UNET features
            if cfg.MODEL.USE_PRETRAINED_UNET:
                # Flatten agent dimension
                B, A, C, H, W = imgs.shape  # Batch, Agents, Channels, Height, Width
                imgs = imgs.view(B * A, C, H, W).to(device)

                # Extract features
                _, _, extracted_features = embedding_extractor(imgs)

                # Reshape back to [B, A, feature_dim]
                extracted_features = extracted_features.view(B, A, -1)
                unet_features = extracted_features.to(device)
            else:
                unet_features = None

            past_loc = past_loc.to(device)
            fut_loc = fut_loc.to(device)

            # Encode observed and future trajectories using ASTRA
            with torch.no_grad():
                x_encoded = astra_model.encode(past_loc, unet_features)
                y_encoded = astra_model.encode(fut_loc, unet_features)

            # Forward pass through CVAE
            mean, log_var, decoded_output = cvae_model(x_encoded, y_encoded)

            # Compute loss
            recon_loss = criterion(decoded_output, x_encoded)
            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
            loss = recon_loss + float(cfg.TRAIN.KL_WEIGHT) * kl_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss.update(loss.item())
            loop.set_postfix(Epoch=epoch+1, Loss=epoch_loss.avg)

        logger.info(f"Epoch [{epoch+1}/{cfg.TRAIN.NUM_EPOCH}] - Loss: {epoch_loss.avg:.5f}")

    # Save trained CVAE model
    torch.save(cvae_model.state_dict(), f'./trained_cvae_weights/{cfg.SUBSET}_cvae_model.pth')
    logger.info("Saved trained CVAE model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CVAE model using pretrained ASTRA and UNET.")
    parser.add_argument("--config_file", default="configs/eth.yaml", type=str, help="Path to config file.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config_file, "r"))
    cfg = munch.munchify(cfg)

    name = create_exp_name(cfg)
    logger.info("Experiment Name: %s", name)
    logger.info("Dataset Name: %s", cfg.DATASET)
    logger.info("Subset Name: %s", cfg.SUBSET)

    train_cvae(cfg)