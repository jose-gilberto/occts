import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sktime.datasets import load_UCR_UEA_dataset
from typing import Optional, Tuple


class UcrDataset(Dataset):
    """Default UcrDataset class used to load the sktime.dataset into
    a generic torch.Dataset format compatible with the models.
    
    Args:
        - X (np.ndarray): series in the correct format to pass to the model.
            The shape has to be (n_instances, n_channels, n_features).
        - y (np.ndarry): The labels that the model will predict. 0 for normal
            class and 1 for anomaly classes.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__()
        assert X.shape[0] == y.shape[0], 'X and y must have the same length (number of instances)'
        self.X = X
        self.y = y
        
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[index]).float(),
            torch.tensor(self.y[index]).float()
        )


def load_ucr_dataset(
    dataset: str,
    verbose: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an UCR dataset from sktime library. The dataset will pass through a pipeline to
    transform the data to the correct format and then pass to a UcrDataset class compatible
    with the torch.DataLoader.
    
    Args:
        - dataset (str): The dataset name from the timeseriesclassification table. Compatible with
            the sktime method to load the data.
    """
    if verbose:
        print(f'Downloading {dataset} dataset from sktime lib...')

    X_train, y_train = load_UCR_UEA_dataset(name=dataset, split='train')
    X_test, y_test = load_UCR_UEA_dataset(name=dataset, split='test')

    # Since the features from the sktime are instatiated as objects we have to manually convert them
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)

    # sktime provides each instance features in a pandas.Series
    X_train_transformed = []
    for val in X_train.values:
        X_train_transformed.append(val[0].tolist())

    X_test_transformed = []
    for val in X_test.values:
        X_test_transformed.append(val[0].tolist())

    # Now X features have a shape like (n_instances, n_features)
    # We need to convert them to a (n_instances, 1, n_features)
    X_train = np.array(X_train_transformed)
    X_train = np.expand_dims(X_train, axis=1)

    X_test = np.array(X_test_transformed)
    X_test = np.expand_dims(X_test, axis=1)
    
    return X_train, X_test, y_train, y_test


def create_datasets(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    positive_class: int,
    train_val_split: Optional[Tuple[float, float]] = None
) -> Tuple[UcrDataset, Optional[UcrDataset], UcrDataset]:
    """
    """
    # Filter only the positive instances used to train the one class models
    X_train_ = X_train[y_train == positive_class]
    y_train_ = y_train[y_train == positive_class]
    
    # Replace all the other classes with a negative label (1) and
    # positive instances with label 0
    y_test_ = [0 if label == positive_class else 1 for label in y_test]
    y_test_ = np.array(y_test_)
    # Instanciate the test dataset since the test data will not change
    # anymore
    test_dataset = UcrDataset(X=X_test, y=y_test_)

    # Split into train and validation sets
    if train_val_split is not None:
        train_size, val_size = train_val_split
        assert train_size + val_size == 1
        # Instanciate the train dataset with the splitted data
        # X = X0 -> Xn being n the result number of the following eq.
        # n = int(number_of_instances * train_split_size)
        train_dataset = UcrDataset(
            X=X_train_[0: int(len(X_train_) * train_size)],
            y=y_train_[0: int(len(X_train_) * train_size)]
        )

        # Instanciate the val dataset with the splitted data
        # X = Xn -> Xm being n the result number of the previous eq.
        # and m the number of instances in the train dataset
        val_dataset = UcrDataset(
            X=X_train_[int(len(X_train_) * train_size):],
            y=y_train_[int(len(X_train_) * train_size):]
        )
        return train_dataset, val_dataset, test_dataset

    train_dataset = UcrDataset(
        X=X_train_, y=y_train_
    )

    return train_dataset, None, test_dataset


def create_dataloaders(
    train_dataset: UcrDataset,
    test_dataset: UcrDataset,
    val_dataset: Optional[UcrDataset] = None,
    batch_size: int = 64,
    drop_last: bool = False
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    """
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )

    val_loader = None

    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last
    )
    
    return train_loader, val_loader, test_loader

