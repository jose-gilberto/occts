import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from typing import Literal, Tuple, Union

from .fcn import FCNClassifier
from .linear import LinearClassifier

__all__ = ['LinearClassifier', 'FCNClassifier']


def get_radius(distance: torch.Tensor, nu: float):
    return np.quantile(np.sqrt(distance.clone().data.cpu().numpy()), 1 - nu)


class OneClassTrainer:
    def __init__(
        self,
        objective: Union[Literal["one-class"], Literal["soft-boundary"]],
        R: torch.Tensor,
        c: torch.Tensor,
        nu: float,
        optimizer_name: str = 'Adam',
        lr: float = 0.001,
        epochs: int = 150,
        lr_milestones: Tuple[int] = [],
        weight_decay: float = 1e-6,
        device: str = 'cuda',
    ) -> None:

        self.optimizer_name = optimizer_name
        self.lr = lr
        self.lr_milestones = lr_milestones
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device

        self.objective = objective
        self.R = torch.tensor(R, device=self.device) # radius R initialized with 0 by default
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 10
        
        # Results
        self.test_auc = None
        self.test_scores = None
        self.train_losses = []

    def init_center(
        self,
        dataloader: DataLoader,
        model: nn.Module,
        eps: float = 0.1
    ) -> torch.Tensor:
        n_samples = 0
        c = torch.zeros(model.latent_dim, device=self.device)
        
        model.eval()
        
        with torch.no_grad():
            for data in dataloader:
                X, _ = data
                X = X.to(self.device)

                outputs = model(X)
                
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
                
        c /= n_samples
        
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        
        return c

    def train(self, dataloader: DataLoader, model: nn.Module) -> None:
        model = model.to(self.device)
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=self.optimizer_name == 'amsgrad'
        )
        
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=self.lr_milestones,
            gamma=0.1
        )
        
        if self.c is None:
            self.c = self.init_center(dataloader, model)
            
        model.train()
        for epoch in range(self.epochs):
            if epoch in self.lr_milestones:
                ...
                
            loss_epoch = 0.0
            n_batches = 0
            # epoch_start_time = 
            
            for data in dataloader:
                X, _ = data
                X = X.to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(X)
                distance = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = distance - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(distance)
                    
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(get_radius(distance, self.nu), device=self.device)
                    
                loss_epoch += loss.item()
                n_batches += 1
            
            self.train_losses.append(loss_epoch / n_batches)
            # print(f'Epoch {epoch + 1}/{self.epochs} - Loss = {loss_epoch / n_batches}')

        return model
    
    def test(self, dataloader: DataLoader, model: nn.Module):
        model = model.to(self.device)
        
        idx_label_score = []
        
        model.eval()
        with torch.no_grad():
            for data in dataloader:
                X, y = data
                X = X.to(self.device)
                
                outputs = model(X)
                
                distance = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = distance - self.R ** 2
                else:
                    scores = distance
                    
                idx_label_score += list(
                    zip(y.cpu().data.numpy().tolist(), scores.cpu().data.numpy().tolist())
                )
                
        self.test_scores = idx_label_score
        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        
        preds = np.maximum(np.zeros_like(scores), scores)
        preds[preds > 0] = 1

        self.test_auc = roc_auc_score(labels, scores)

        print(f'\t\t\tTest set AUC {self.test_auc}', )
        print(f'\t\t\tTest set Accuracy {accuracy_score(labels, preds)}')

