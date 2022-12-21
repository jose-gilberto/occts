import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple, List


def fit_autoencoder(
    auto_encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    device: torch.cuda.device,
    save_weights: bool = False,
    save_dir: str = '.'
) -> Tuple[nn.Module, Dict[str, List[float]]]:

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    optimizer = torch.optim.Adam(
        auto_encoder.parameters(),
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    criterion = nn.MSELoss()

    history = {
        'train': [],
        'val': [],
    }

    pbar = tqdm(range(1, epochs + 1, 1))

    for epoch in pbar:
        pbar.set_description(f'Epoch {epoch}')

        auto_encoder.train()
        train_losses = []

        for x, _ in train_loader:
            x = x.to(device)

            optimizer.zero_grad()

            x_hat = auto_encoder(x)

            loss = criterion(x_hat, x)
            loss.backward()

            optimizer.step()

            train_losses.append(loss.item())

        auto_encoder.eval()
        val_losses = []

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)

                x_hat = auto_encoder(x)

                loss = criterion(x_hat, x)

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        pbar.set_postfix_str(
            f'Train Loss = {round(train_loss, 5)}, Val Loss = {round(val_loss, 5)}'
        )

    if save_weights:
        torch.save({
            'auto_encoder_dict': auto_encoder.state_dict()
        }, f'{save_dir}/weights/auto_encoder_parameters.pth')

    return auto_encoder, history
