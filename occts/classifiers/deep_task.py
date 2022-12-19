from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple
from .deep import OneClassTrainer


class OneClassTask:
    def __init__(self, model: nn.Module, objective: str = 'one-class', nu: float = 0.1):
        assert objective in ('one-class', 'soft-boundary')
        self.objective = objective
        assert 0 < nu and nu <= 1
        self.nu = nu
        self.R = 0    # Hypersphere radius
        self.c = None # Hypersphere center
        
        self.model = model
        
        self.trainer = None
        self.optimizer = None
        
        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None
        }
        
    def train(
        self,
        dataloader: DataLoader,
        optimizer: str = 'adam',
        lr: float = 0.001,
        epochs: int = 50,
        lr_milestones: Tuple[int] = [],
        weight_decay: float = 1e-6,
        device: str = 'cuda',
    ):
        self.trainer = OneClassTrainer(
            objective=self.objective,
            R=self.R,
            c=self.c,
            nu=self.nu,
            optimizer_name=optimizer,
            lr=lr,
            epochs=epochs,
            lr_milestones=lr_milestones,
            weight_decay=weight_decay,
            device=device,
        )

        self.model = self.trainer.train(dataloader, self.model)
        self.R = float(self.trainer.R.cpu().data.numpy())
        self.c = self.trainer.c.cpu().data.numpy().tolist()
        
    def test(self, dataloader: DataLoader, device: str = 'cuda'):
        if self.trainer is None:
            self.trainer = OneClassTrainer(
                self.objective, self.R, self.c, self.nu, device=device
            )
            
        self.trainer.test(dataloader, self.model)
        
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_scores'] = self.trainer.test_scores
