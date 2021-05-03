import copy
from galaxyquest.cnn import EFIGIDataset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pickle
import torch
from torch import Tensor
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

class Model:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.dataset_size = len(self.dataset)

    def get_dataloaders(self, test_size, val_size, batch_size, num_workers):
        # Generate a stratified train/test split using labels from the dataset object.
        # Train split
        first_split = StratifiedShuffleSplit(n_splits=1, test_size=(test_size + val_size))
        (train_index, split_index) = next(first_split.split(np.zeros(self.dataset_size), self.dataset.get_labels()))
        second_split = StratifiedShuffleSplit(n_splits=1, test_size=(test_size / (test_size + val_size)))
        second_split_size = len(split_index)
        (test_index, val_index) = next(second_split.split(np.zeros(second_split_size), self.dataset.get_labels(split_index)))


        # Determine the number of samples for our different sets
        num_train_samples = len(train_index)
        num_test_samples = len(test_index)
        num_val_samples = len(val_index)


        self.dataset_sizes = {
            "train": num_train_samples,
            "test": num_test_samples,
            "val": num_val_samples
        }

        # These sampler objects get passed to the dataloader with our training and testing indices,
        # to split the data accordingly.
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_index)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_index)

        DataLoaders = {
            "train": torch.utils.data.DataLoader(self.dataset, batch_size = batch_size, num_workers = num_workers, sampler = train_sampler),
            "val": torch.utils.data.DataLoader(self.dataset, batch_size = batch_size, num_workers = num_workers, sampler = val_sampler),
            "test": torch.utils.data.DataLoader(self.dataset, batch_size = batch_size, num_workers = num_workers, sampler = test_sampler)
        }

        return DataLoaders


    def train_model(self, num_epochs=25):
        dataloaders = self.get_dataloaders(.15, .15, 4, 4)
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = nn.CrossEntropyLoss() 
        

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.model.to(device)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        epoch_metrics = {
            "epoch_acc":  [],
            "epoch_loss": [],
            "predicted_labels": [],
            "prediction_probabilities": [],
            "prediction_confidences": [],
            "ground_truth_labels": [],
            "pgc_ids": []
        }

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val', 'test']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                if phase == "test":
                    epoch_metrics["predicted_labels"].append([])
                    epoch_metrics["prediction_probabilities"].append([])
                    epoch_metrics["ground_truth_labels"].append([])
                    epoch_metrics["pgc_ids"].append([])
                    epoch_metrics["prediction_confidences"].append([])

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for sample in dataloaders[phase]:
                    
                    inputs = sample["image"].to(device)
                    labels = sample["label"].to(device)
                    pgc_ids = sample["pgc_id"]

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        confidence, preds = torch.max(outputs, 1)

                        # Normalize output scores to a probability.
                        confidence = nn.functional.softmax(confidence, dim=0)
                        
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += int(torch.sum(preds == labels.data))

                    if phase == "test":
                        epoch_metrics["ground_truth_labels"][epoch].extend(labels.flatten().tolist())
                        epoch_metrics["predicted_labels"][epoch].extend(preds.flatten().tolist())
                        epoch_metrics["pgc_ids"][epoch].extend(pgc_ids)
                        epoch_metrics["prediction_confidences"][epoch].extend(confidence.flatten().tolist())

                if phase == 'train':
                    scheduler.step()

                epoch_loss = float(running_loss) / self.dataset_sizes[phase]
                epoch_acc = float(running_corrects) / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model if it's the best accuracy seen so far
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                # Evaluate the model against the test set in the test phase
                if phase == 'test':
                    epoch_metrics["epoch_acc"].append(epoch_acc)
                    epoch_metrics["epoch_loss"].append(epoch_loss)

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        # Save the trained model.
        self.save_model()

        # Save the evaluation results.
        self.eval_results = epoch_metrics
        self.save_eval_results()
        
    def save_model(self, out_path = "model.pt"):
        torch.save(self.model, out_path)

    def save_eval_results(self, out_path = "cnn_eval.p"):
        with open(out_path, "wb") as out_file:
            pickle.dump(self.eval_results, out_file)
 