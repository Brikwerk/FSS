import pickle

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from skimage import io, img_as_ubyte
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm

from meta_mae import MaskedAutoencoderViT


def to_patches(imgs, patch_size):
    num_patches = imgs.shape[2] // patch_size
    num_channels = imgs.shape[1]
    x = imgs.reshape(imgs.shape[0], 3, num_patches, patch_size, num_patches, patch_size)
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(imgs.shape[0], num_patches ** 2, patch_size ** 2 * num_channels)
    return x


def to_imgs(x, patch_size, num_patches, num_channels):
    n_p = num_patches
    n_c = num_channels
    p_s = patch_size
    # Reshape last dim to (patch_size, patch_size, num_channels)
    # Tensor should be in the shape of (batch, num_patches_h, num_patches_w, patch_size, patch_size, num_channels)
    x = x.reshape(x.shape[0], n_p, n_p, p_s, p_s, n_c)
    # Rearrange dimensions to enable image reshaping
    x = torch.einsum('nhwpqc->nchpwq', x)
    # Reshape into an image
    x = x.reshape(x.shape[0], n_c, n_p * p_s, n_p * p_s)
    return x


class MINDataset(Dataset):
    def __init__(self, min_path, img_size=84):
        with open(min_path, 'rb') as f:
            min_raw_data = pickle.load(f)
        
        self.image_data = min_raw_data['image_data']
        self.label_data = np.zeros(len(self.image_data))
        for i, class_label in enumerate(min_raw_data['class_dict'].keys()):
            idxs = min_raw_data['class_dict'][class_label]
            self.label_data[idxs] = i

        self.transform = T.Compose([
            T.ToTensor(),
            # transforms.Normalize(mean=[0.4707, 0.4495, 0.4026],
            #                      std=[0.2843, 0.2752, 0.2903])
            T.RandomResizedCrop((img_size, img_size))
        ])

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, idx):
        img = self.image_data[idx]
        img = self.transform(img)

        label = torch.tensor(self.label_data[idx]).float()
        return {
            'image': img,
            'label': label
        }


if __name__ == '__main__':
    # Traing Settings
    WARMUP_EPOCHS = 40
    EPOCHS = 800
    BASE_LR = 1.5e-4
    BATCH_SIZE = 1024
    ACCUM_BATCH_SIZE = 4096
    ACCUM_ITER = ACCUM_BATCH_SIZE // BATCH_SIZE
    LR = BASE_LR * ACCUM_BATCH_SIZE / 256
    IMG_SIZE = 84

    # Model Settings
    PATCH_SIZE = 6
    NUM_HEADS = 16
    ENCODER_DEPTH = 4
    DECODER_DEPTH = 4
    EMBED_DIM = 384
    DECODE_DIM = 512

    train_dataset = MINDataset("/Datasets/mini-imagenet-l2l/mini-imagenet-cache-train.pkl")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = MaskedAutoencoderViT(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_heads=NUM_HEADS,
        depth=ENCODER_DEPTH,
        embed_dim=EMBED_DIM,
        decoder_num_heads=NUM_HEADS,
        decoder_depth=DECODER_DEPTH,
        decoder_embed_dim=DECODE_DIM
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, 
        warmup_epochs=WARMUP_EPOCHS*len(train_loader), max_epochs=EPOCHS*len(train_loader))
    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()

    for epoch in range(EPOCHS):
        output = None
        batch = None
        losses = []
        for i, batch in enumerate(tqdm(train_loader)):
            batch = {key: value.to(device) for key, value in batch.items()}

            with torch.cuda.amp.autocast():
                loss, output, mask = model(batch['image'])
        
            losses.append(loss.item())

            scaler.scale(loss).backward()

            if ((i + 1) % ACCUM_ITER == 0):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
            scheduler.step()
        
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
            
        print(f'Epoch {epoch}: Loss {np.mean(losses)}')
        writer.add_scalar("train/loss", np.mean(losses), epoch)

        # Save an image from the output batch
        output = to_imgs(output, PATCH_SIZE, IMG_SIZE // PATCH_SIZE, 3)
        recon_img = output[0].detach().cpu().numpy()
        # Conver to uint8-space for a regular image
        # recon_img = (np.transpose(recon_img, (1, 2, 0))*255).astype(np.uint8)
        # io.imsave('train_img.png', recon_img) # Save
        writer.add_image('reconstruction', recon_img, epoch)

        orig_img = batch['image'][0].cpu().numpy()
        # orig_img = T.ToPILImage()(batch['image'][0])
        # orig_img.save("train_orig_img.png")
        writer.add_image('original', orig_img, epoch)

        state_dict = {
            "model": model.state_dict(),
            "model_settings": {
                "WARMUP_EPOCHS": WARMUP_EPOCHS,
                "EPOCHS": EPOCHS,
                "BASE_LR": BASE_LR,
                "BATCH_SIZE": BATCH_SIZE,
                "ACCUM_BATCH_SIZE": ACCUM_BATCH_SIZE,
                "ACCUM_ITER": ACCUM_ITER,
                "LR": LR,
                "IMG_SIZE": IMG_SIZE,
                "PATCH_SIZE": PATCH_SIZE,
                "NUM_HEADS": NUM_HEADS,
                "ENCODER_DEPTH": ENCODER_DEPTH,
                "DECODER_DEPTH": DECODER_DEPTH,
                "EMBED_DIM": EMBED_DIM,
                "DECODE_DIM": DECODE_DIM
            }
        }

        torch.save(state_dict, f'./fit_models/mae_pretrain.pt')
