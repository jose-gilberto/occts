""""""
import numpy as np
import yaml
from yaml.loader import SafeLoader

from occts.classifiers.deep import FCNClassifier
from occts.classifiers.deep_task import OneClassTask
from occts.utils.data import load_ucr_dataset, create_datasets, create_dataloaders


experiments_variables = yaml.


# Binary datasets
DATASETS = [
    'Yoga', 'WormsTwoClass', 'Wine', 'Wafer', 'TwoLeadECG', 'Strawberry', 'SemgHandGenderCh2', 'BeetleFly',
    'BirdChicken', 'Computers', 'DistalPhalanxOutlineCorrect', 'Earthquakes', 'ECG200', 'ECGFiveDays',
    'FordA', 'FordB', 'HandOutlines', 'ItalyPowerDemand', 'MiddlePhalanxOutlineCorrect', 'Chinatown',
    'FreezerRegularTrain', 'FreezerSmallTrain', 'GunPointAgeSpan', 'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung', 'PowerCons', 'Coffee', 'Ham', 'Herring', 'Lightning2', 'MoteStrain',
    'PhalangesOutlinesCorrect', 'ProximalPhalanxOutlineCorrect', 'ShapeletSim', 'SonyAIBORobotSurface1',
    'SonyAIBORobotSurface2', 'ToeSegmentation1', 'ToeSegmentation2', 'HouseTwenty'
]

dataset_list = []
dataset_label = []
dataset_roc_auc = []
dataset_acc = []

for dataset in DATASETS:
    print(f'Testing with dataset: {dataset}...')
    x_train, x_test, y_train, y_test = load_ucr_dataset(dataset=dataset)

    # Get unique labels to do the one class classification task
    unique_labels = np.unique(y_train)

    for label in unique_labels:
        print(f'\tClassifying the label {label}...')

        train_dataset, _, test_dataset = create_datasets(x_train, x_test, y_train, y_test, label)
        train_loader, _, test_loader = create_dataloaders(train_dataset=train_dataset, test_dataset=test_dataset)

        model = FCNClassifier(in_channels=1, latent_dim=32)
        octask = OneClassTask(model=model, objective='soft-boundary', nu=0.1)

        octask.train(dataloader=train_loader, epochs=150, lr=1e-3, weight_decay=1e-6)
        octask.test(dataloader=test_loader)

        dataset_list.append(dataset)
        dataset_label.append(label)
        dataset_acc.append(octask.trainer.test_acc)
        dataset_roc_auc.append(octask.trainer.test_auc)

