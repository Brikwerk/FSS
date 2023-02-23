import os
import argparse

import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as T

from tqdm import tqdm

from meta_mae import mae_vit_base_patch16, MaskedAutoencoderViT
from src.data.fs_sampler import FewShotSampler
from src.data.min_dataset import MINDataset
from src.utils import pairwise_distances_logits, accuracy
from src.data.isic import ISICDataset


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def reload_model(device):
    model = mae_vit_base_patch16()
    checkpoint = torch.load("./fit_models/mae_visualize_vit_base.pth")
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    return model


def mae_forward(model, x):
    B = x.shape[0]
    x = model.patch_embed(x)

    cls_tokens = model.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + model.pos_embed
    # x = model.pos_drop(x)

    for blk in model.blocks:
        x = blk(x)

    # if self.global_pool:
    #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
    #     outcome = self.fc_norm(x)
    # else:
    x = model.norm(x)
    outcome = x[:, 0]

    return outcome


parser = argparse.ArgumentParser()
# Config
parser.add_argument('--root_data_path', required=True, type=str,
                    help="""Path to the root data folder. Must be the
                            parent folder containing dataset folders.""")
# parser.add_argument('--datasets', required=False, type=str, nargs='+',
#                     choices=["min", "bccd", "hep",
#                              "chestx", "isic", "eurosat", "plant", "ikea"],
#                     help="Choose a subset of datasets to test")
parser.add_argument('--dataset', required=True, type=str,
                    choices=["min", "bccd", "hep",
                             "isic", "eurosat", "plant"],
                    help="Choose a datasets to test")
parser.add_argument('--test_episodes', required=True, type=int,
                    help="""Number of episodes to test.""")
# Hyperparams
parser.add_argument('--pretrain_epochs', required=False, default=10, type=int,
                    help="""Number of MAE pretraining epochs per episode.""")
parser.add_argument('--pretrain_iters', required=False, default=50, type=int,
                    help="""Number of MAE pretraining iterations per episode.""")
parser.add_argument('--finetune_epochs', required=False, default=1, type=int,
                    help="""Number of MAE finetuning epochs per episode.""")
parser.add_argument('--finetune_iters', required=False, default=4, type=int,
                    help="""Number of MAE finetuning iterations per episode.""")
# Few-shot params
parser.add_argument('--shots', required=True, type=int,
                    help="""Number of labelled examples in an episode""")
parser.add_argument('--ways', required=False, default=5, type=int,
                    help="""Number of classes used in an episode.""")
parser.add_argument('--query', required=False, default=15, type=int,
                    help="""Number of unlabelled examples in an episode.""")
parser.add_argument('--img_size', required=True, type=int,
                    help="""Image size used.""")
# Model
# parser.add_argument('--model_type', default=False, required=True, choices=[
#                         'CONV4', "WRN", "RESNET12", "RESNET18",
#                         "RESNET50", "DINO_SMALL", "CONV4_BASE",
#                         "CONV4_VANILLA"
#                     ], help="""Model type. Either CONV4, 
#                     RESNET18, or DINO_SMALL""")
# parser.add_argument('--model_path', default=False, required=True, type=str,
#                     help="Path to model weights")
parser.add_argument('--device', default='cuda', type=str,
                    help="Device to use for testing (ex: cuda:0).")
args = parser.parse_args()


def get_sampler(dataset_name, root, img_size, ways, shots, query):
    # IN1K mean/std
    mean = torch.tensor([0.4707, 0.4495, 0.4026])
    std = torch.tensor([0.2843, 0.2752, 0.2903])

    sampler = None
    dataset = None
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
        T.Resize(size=(img_size, img_size)),
    ])
    if dataset_name == "min":
        dataset = torchvision.datasets.ImageFolder(
            root=os.path.join(root, "mini-imagenet", "test"), transform=test_transform)
    elif dataset_name == "isic":
        dataset = ISICDataset(os.path.join(root, "isic2018"), transform=test_transform)
    elif dataset_name == "eurosat":
        dataset = torchvision.datasets.ImageFolder(os.path.join(root, "eurosat"), transform=test_transform)
    elif dataset_name == "plant":
        dataset = torchvision.datasets.ImageFolder(os.path.join(root, "plant", "test"), transform=test_transform)
    elif dataset_name == "hep":
        dataset = torchvision.datasets.ImageFolder(os.path.join(root, "HEp-Dataset"), transform=test_transform)
    elif dataset_name == "bccd":
        dataset = torchvision.datasets.ImageFolder(os.path.join(root, "wbc-aug"), transform=test_transform)
    sampler = FewShotSampler(dataset, WAYS, SHOTS, QUERY)

    return sampler


if __name__ == "__main__":
    # Testing setting
    WAYS = args.ways
    SHOTS = args.shots
    QUERY = args.query
    IMG_SIZE = args.img_size
    TEST_EPISODES = args.test_episodes

    # Hyperparams
    EPOCHS = args.pretrain_epochs
    ITERATIONS = args.pretrain_iters
    CE_EPOCHS = args.finetune_epochs
    CE_ITERATIONS = args.finetune_iters
    BASE_LR = 1.5e-4
    BATCH_SIZE = 100
    ACCUM_BATCH_SIZE = 2500
    ACCUM_ITER = ACCUM_BATCH_SIZE // BATCH_SIZE
    # LR = BASE_LR * ACCUM_BATCH_SIZE / 256
    LR = 1e-5 # For MAE SS pretraining
    LR_CE = 1e-4 # For MAE CE support finetuning

    test_sampler = get_sampler(
        dataset_name=args.dataset,
        root=args.root_data_path,
        img_size=IMG_SIZE,
        ways=WAYS,
        shots=SHOTS,
        query=QUERY
    )

    # Model creation and load
    device = args.device
    model = mae_vit_base_patch16()
    checkpoint = torch.load("./fit_models/mae_visualize_vit_base.pth")
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)

    # Testing loop
    test_losses = []
    test_accuracies = []
    for episode in range(TEST_EPISODES):
        scaler = torch.cuda.amp.GradScaler()
        model = reload_model(device)

        # Sample Episodic batch
        data, labels = test_sampler.get_batch()
        data, labels = data.to(device).squeeze(0), labels.to(device).squeeze(0)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.05)
        loss_mse = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            train_losses = []
            for i in range(ITERATIONS):
                aug_data = T.RandomResizedCrop((224,224))(data)
                with torch.cuda.amp.autocast():
                    loss, _, _ = model(aug_data)
                
                train_losses.append(loss.item())

                scaler.scale(loss).backward()

                # if ((i + 1) % ACCUM_ITER == 0):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
                # scheduler.step()

            print(f"Epoch {epoch} | Loss: {np.mean(train_losses):.4f}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_CE, betas=(0.9, 0.95), weight_decay=0.05)
        loss_ce = nn.CrossEntropyLoss()

        for epoch in range(CE_EPOCHS):
            losses = []
            accuracies = []
            for i in tqdm(range(CE_ITERATIONS)):
                sort = torch.sort(labels)
                # data = data.squeeze(0)[sort.indices].squeeze(0)
                labels = labels.squeeze(0)[sort.indices].squeeze(0)

                b, c, h, w = data.size()
                support_batch = data.reshape(WAYS, (QUERY + SHOTS), c, h, w)
                support_batch = support_batch[:, :5]
                repeat_count = (QUERY + SHOTS) // SHOTS
                if (QUERY + SHOTS) % SHOTS > 0:
                    support_batch = support_batch.repeat(1, repeat_count+1, 1, 1, 1)
                    support_batch = support_batch[:, :(QUERY + SHOTS)]
                else:
                    support_batch = support_batch.repeat(1, repeat_count, 1, 1, 1)

                # support_batch = support_batch[torch.randperm(WAYS), :, :]
                # support_batch = support_batch[:, torch.randperm(SHOTS + QUERY), :]

                support_batch = support_batch.reshape(WAYS * (QUERY + SHOTS), c, h, w)

                with torch.cuda.amp.autocast():
                    prototypes = mae_forward(model, support_batch)
                    # prototypes = model_vit.forward_features(support_batch)

                    # a = prototypes.reshape(WAYS, (SHOTS + QUERY), -1)
                    # b = a[torch.randperm(WAYS), :, :]
                    # b = b[:, torch.randperm(SHOTS + QUERY), :]
                    # prototypes = ((a + b)/2).reshape(WAYS * (SHOTS + QUERY), -1)

                    support_indices = np.zeros(data.size(0), dtype=bool)
                    selection = np.arange(WAYS) * (SHOTS + QUERY)
                    for offset in range(SHOTS):
                        support_indices[selection + offset] = True
                    query_indices = torch.from_numpy(~support_indices)
                    support_indices = torch.from_numpy(support_indices)

                    # # Use support from prototypes
                    support = prototypes.squeeze(0)[support_indices]
                    support = support.reshape(WAYS, SHOTS, -1)
                    support = support.mean(dim=1).squeeze(0)

                    query = prototypes.squeeze(0)
                    query = query[query_indices]

                    # Calculate accuracy
                    logits = pairwise_distances_logits(query, support)
                    query_labels = labels[query_indices].long()
                    acc = accuracy(logits.squeeze(0), query_labels.squeeze(0))
                    accuracies.append(acc.item())
                    
                    # Calculate loss
                    loss = loss_ce(logits.squeeze(0), query_labels.squeeze(0))

                    # loss.backward()
                    scaler.scale(loss).backward()

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    # scheduler.step()

                losses.append(loss.item())
                
            print(f'Epoch {epoch}: CE Loss {np.mean(losses)} FS Accuracy {np.mean(accuracies)}')

        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        with torch.no_grad():
            prototypes = mae_forward(model, data)
            # prototypes = model_vit.forward_features(data)

        support_indices = np.zeros(data.size(0), dtype=bool)
        selection = np.arange(WAYS) * (SHOTS + QUERY)
        for offset in range(SHOTS):
            support_indices[selection + offset] = True
        query_indices = torch.from_numpy(~support_indices)
        support_indices = torch.from_numpy(support_indices)

        # Use average of reconstructed examples in the output
        support = prototypes.squeeze(0)[support_indices]
        support = support.reshape(WAYS, SHOTS, -1)
        support = support.mean(dim=1).squeeze(0)

        query = prototypes.squeeze(0)
        query = query[query_indices]

        # Calculate accuracy
        logits = pairwise_distances_logits(query, support)
        labels_query = labels[query_indices].long()

        acc = accuracy(logits.squeeze(0), labels_query.squeeze(0))
        test_accuracies.append(acc.item())
        loss = loss_ce(logits.squeeze(0), labels_query.squeeze(0))
        test_losses.append(loss.item())
        
        print("---")
        print(f'Test {episode}: Loss {loss.item()} Accuracy {acc.item()}')
        print("---")
    
    print("===")
    print(f'Overall: Loss {np.mean(test_losses)} Accuracy {100*np.mean(test_accuracies):.3f}')
    print("===")
