import torch
from torch import nn
from collections import OrderedDict
import torch.nn as nn
import os
import json
from torch.utils.data import DataLoader
import copy
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import yaml
import torchvision.transforms.v2 as T
from torchvision.datasets import GTSRB
import pickle
import random


class ActivationFunction:
    def __init__(self):
        self._activations = {
            'relu': nn.ReLU,
            'lrelu': nn.LeakyReLU,
            'elu': nn.ELU,
            'selu': nn.SELU,
            'gelu': nn.GELU,
            'swish': nn.SiLU, 
        }
    def __call__(self, name: str, **kwargs):
        name = name.lower()
        if name not in self._activations:
            raise ValueError("[ERROR] Selected activation not allowed")
        
        return self._activations[name](**kwargs)

class Flatten:
    def __init__(self):
        self.flatten_layer = nn.Flatten()
    def __call__(self, input):
        return self.flatten_layer(input)
    
class Encoder(nn.Module):
    # hidden_layer_configs: list
    # a list whose elements are tuples with as many elements as the number of hidden layers {<key>, <value>} 
    # where key is the number of neurons 
    # and value is the activation function
    # this way the model will be able to use different activation function at different depth
    def __init__(self, input_layer_size: int, hidden_layer_configs: list, latent_space_dim: int):
        super().__init__()
        self.input_layer_size = input_layer_size
        self.model = nn.Sequential()

        self.model.add_module("flatten", nn.Flatten())

        # Setting the input layer size as the current
        current_input_size = input_layer_size
        activation = ActivationFunction()

        for i, (current_layer_neurons_cardinality, current_layer_activation_function) in enumerate(hidden_layer_configs):
            current_linear = nn.Linear(current_input_size, current_layer_neurons_cardinality)
            self.model.add_module("hidden_{}".format(i), current_linear)
            activation_function = activation(current_layer_activation_function)
            self.model.add_module("activation_{}".format(i), activation_function) 
            # for the next layer I have to re-assign the outputsize of the previous (i.e: the input size of the current)
            current_input_size = current_layer_neurons_cardinality

        self.latent_space_dim = latent_space_dim
        self.model.add_module("to_latentspace_layer", nn.Linear(current_input_size, latent_space_dim))

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_space_dim: int, hidden_layer_configs: list, output_layer_size: int):
        super().__init__()
        self.latent_space_dim = latent_space_dim
        self.model = nn.Sequential()
        current_input_size = latent_space_dim
        activation = ActivationFunction()
        for i, (current_layer_neurons_cardinality, current_layer_activation_function) in enumerate(hidden_layer_configs[::-1]):
            current_linear = nn.Linear(current_input_size, current_layer_neurons_cardinality)
            self.model.add_module("hidden_{}".format(i), current_linear)
            activation_function = activation(current_layer_activation_function)
            self.model.add_module("activation_{}".format(i), activation_function) 
            current_input_size = current_layer_neurons_cardinality
        
        self.out_space_dim = output_layer_size
        self.model.add_module("to_origspace_layer", nn.Linear(current_input_size, self.out_space_dim))

    def forward(self, x):
        return self.model(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_layer_size: int, hidden_layer_configs: list, latent_space_dim: int):
        super().__init__()
        self.encoder = Encoder(input_layer_size, hidden_layer_configs, latent_space_dim)
        self.decoder = Decoder(latent_space_dim, hidden_layer_configs, input_layer_size)
    
    def forward(self, x):
        # X --> Z --> Y
        z = self.encoder(x)
        y = self.decoder(z)
        return y
    
def set_deterministic_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

class Experiment:
    def __init__(self, training_data, testing_data,
                       title: str, motivation: str, 
                       directory_name: str, base_dir: str = "experiments", 
                       need_valset: bool = False, valset_proportion: float = None,
                       learning_rate=None, optim=None, batch_size=None, num_epochs=None):
        
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.title = title
        self.motivation = motivation
        self.base_dir = base_dir
        self.directory_name = directory_name
        self.experiment_path = os.path.join("../", self.base_dir, self.directory_name)
        self.training_data = training_data 
        self.testing_data = testing_data   
        self.valset_proportion = valset_proportion
        self.need_valset = need_valset
        self.learning_rate = learning_rate
        self.optim = optim
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # if an experiment is created on the assumption that it requires a validation set
        if self.need_valset:
            # but the valset proportion is not provided or invalid
            if self.valset_proportion is None or self.valset_proportion <= 0 or self.valset_proportion >= 1:
                raise ValueError("[ERROR] valset_proportion argument must be in (0,1)")
            
            # otherwise we consider the total length of the training data
            total_train_size = len(self.training_data)
            # compute the size of the validation set given the argument proportion in the ctor
            val_size = int(total_train_size * self.valset_proportion)

            original_training_sample_indices = []
            original_training_sample_classes = []

            # for each index and related sample in training data
            for i, tup in enumerate(self.training_data._samples):
                # populate the two lists
                original_training_sample_indices.append((i))
                original_training_sample_classes.append(tup[1])

            # Apply a stratified sampling over those indices, stratifying on the related classes 
            # (this maintains proportions between classes) 
            TRAIN_INDICES, VAL_INDICES = train_test_split(original_training_sample_indices,
                                                              stratify=original_training_sample_classes, 
                                                              test_size=val_size)
            # creating a deep copy of the original data
            self.validation_set = copy.deepcopy(self.training_data)
            # keeping in this copy only those samples whose index is in the validation indices list
            self.validation_set._samples = [self.training_data._samples[i] for i in VAL_INDICES]
            # same thing but for training data
            self.training_data._samples = [self.training_data._samples[i] for i in TRAIN_INDICES]

        if not os.path.exists(self.experiment_path):
                os.makedirs(self.experiment_path, exist_ok=True)
    
    # Connecting the current experiment to the model we want to use
    def connect_model(self, model):
        self.model = model.to(self.device)

    # preparing dataloaders
    def allocate_dataloaders(self, in_batch_size: int):
        self.train_loader = DataLoader(self.training_data, batch_size=in_batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testing_data, batch_size=in_batch_size, shuffle=False)
        if self.need_valset:
            self.validation_loader = DataLoader(self.validation_set, batch_size=in_batch_size, shuffle=False)

    def train_one_epoch(self, dataloader, optimizer, verbose=False):
        self.model.train()
        batch_losses = []
        num_batches = len(dataloader)
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            if verbose:
                print(">> Training on batch: {}/{}".format(i+1, num_batches))
            optimizer.zero_grad()
            y_hat = self.model(x) # X --> Z --> Y
            batch_loss = F.mse_loss(y_hat, x.view(x.size(0), -1))
            batch_loss.backward()
            optimizer.step()
            batch_losses.append(batch_loss.item())
        return np.mean(batch_losses)

    def evaluate(self, dataloader, verbose=False):
        self.model.eval()
        batch_losses = []
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                x = x.to(self.device)
                y_hat = self.model(x)
                target = x.view(x.size(0), -1)
                batch_loss = F.mse_loss(y_hat, target)
                batch_losses.append(batch_loss.item())
                if verbose:
                    print(f"> Evaluating batch: {i}/{num_batches}")

        epoch_loss = np.mean(batch_losses)
        return epoch_loss
    

    def train(self, num_epochs: int = None, learning_rate=None, optim = None, 
              early_stopping: bool = False, patience: int = np.inf, max_iterations: int = 2000, verbose: int = 1):
        
        if num_epochs is None:
            num_epochs = self.num_epochs
        if learning_rate is None:
            learning_rate = self.learning_rate
        if optim is None:
            optim = self.optim

        if early_stopping and patience == np.inf:
            raise ValueError("[ERROR] Cannot define early stopping without providing a valid patience argument.")

        if optim is None:
            print("[WARNING] No optimizer provided, setting Adam default")
            opt = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            opt = optim

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.opt_to_str = str(type(opt))
        
        self.epoch_losses = [] 
        if self.need_valset:
            self.validation_losses = [] 
            lowest_validation_loss = np.inf
            no_better_validation_count = 0

        print("Training started on {} total epochs with learning_rate {}".format(num_epochs, learning_rate))

        for epoch_idx in range(num_epochs):
            if epoch_idx < max_iterations:
                if verbose == 2 or verbose == 1:
                    print("---"*10)
                    print("> Training epoch {}/{}".format(epoch_idx+1, num_epochs))
                if verbose == 2:
                    epoch_loss = self.train_one_epoch(self.train_loader, opt, verbose=True)
                else:
                    epoch_loss = self.train_one_epoch(self.train_loader, opt)
                
                self.epoch_losses.append(epoch_loss)
                
                print("> Epoch: {}/{} Loss: {}".format(epoch_idx+1, num_epochs, epoch_loss))
                if self.need_valset:
                    if verbose == 2:
                        val_epoch_loss = self.evaluate(self.validation_loader, verbose=True)
                    else:
                        val_epoch_loss = self.evaluate(self.validation_loader)
                    self.validation_losses.append(val_epoch_loss)
                    
                    print("> Epoch: {}/{} Validation Loss: {}".format(epoch_idx+1, num_epochs, val_epoch_loss))
                    if early_stopping:
                        if val_epoch_loss < lowest_validation_loss:
                            lowest_validation_loss = val_epoch_loss
                            no_better_validation_count = 0
                            model_path = os.path.join(self.experiment_path, "{}_model.pth".format(self.title))
                            torch.save(self.model.state_dict(), model_path)
                        else:
                            no_better_validation_count += 1
                            if no_better_validation_count >= patience:
                                print("[EARLY STOPPING] No validation loss decreasing after {} iterations.\n>>Training stopped.".format(patience))
                                break
            else:
                print("[WARNING] Training interrupted for reaching max_iterations")
                break

    def save_results(self, test_metrics: dict):
        model_path = os.path.join(self.experiment_path, "{}_last_model.pth".format(self.title))
        torch.save(self.model.state_dict(), model_path)
        results = {
            "title": self.title,
            "motivation": self.motivation,
            "hyperparams":{
                "learning_rate": self.learning_rate,
                "optimizer": self.opt_to_str,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size
            },
            "test_metrics": {
                "loss": float(test_metrics["loss"])
            },
            "training_losses": [float(x) for x in self.epoch_losses],
            "validation_losses": [float(x) for x in self.validation_losses] if self.need_valset else []
        }
        
        with open(os.path.join(self.experiment_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_losses, label='Training Loss', color='blue')
        if self.need_valset:
            plt.plot(self.validation_losses, label='Validation Loss', color='orange')
        plt.title("Training Loss - {}".format(self.title))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.experiment_path, "loss_chart.png"))
        plt.close()
        
        print("Results saved at: {}".format(self.experiment_path))

    def start_experiment(self, model, num_epochs=None, in_batch_size=None, learning_rate=None, 
                         optim=None, early_stopping=False, patience=10, max_iterations=2000, verbose=1):
        
        if in_batch_size is None:
            in_batch_size = self.batch_size
        if num_epochs is None:
            num_epochs = self.num_epochs
        if learning_rate is None:
            learning_rate = self.learning_rate
        if optim is None:
            optim = self.optim

        self.batch_size = in_batch_size
        self.connect_model(model)
        self.allocate_dataloaders(in_batch_size)
        
        self.train(num_epochs=num_epochs, learning_rate=learning_rate, optim=optim, 
                   early_stopping=early_stopping, patience=patience, 
                   max_iterations=max_iterations, verbose=verbose)
        
        model_path = os.path.join(self.experiment_path, "{}_model.pth".format(self.title))
        if early_stopping and os.path.exists(model_path):
            print("> Retrieving best model params")
            self.model.load_state_dict(torch.load(model_path))

        test_loss = self.evaluate(self.test_loader)
        test_results = {"loss": test_loss}
        self.save_results(test_results)
        print(">>> Experiment Ended")
    

# Helpers functions

def get_config():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def unpack_config(config) -> dict:
    input_space_dim = config['INPUT_SPACE_DIMENSION']
    latent_space_dim = config['LATENT_SPACE_DIMENSION']
    bs = config['TRAINING']['BATCH_SIZE']
    lr = config['OPTIMIZER']['LR']
    wd = config['OPTIMIZER']['DECAY']
    num_epochs = config['TRAINING']['NUM_EPOCHS']
    neurons_cardinality_per_layer = config['AUTOENCODER_CONFIG']['NEURONS']
    activations_per_layer = config['TRAINING']['ACTIVATION_FUNCTIONS']
    possible_activations = list(product(activations_per_layer, repeat=len(neurons_cardinality_per_layer)))
    config_as_dict = {
        'input_space_dim': input_space_dim,
        'latent_space_dim': latent_space_dim,
        'bs': bs,
        'lr': lr,
        'wd': wd,
        'num_epochs':num_epochs,
        'neurons_cardinality_per_layer': neurons_cardinality_per_layer,
        'possible_activations': possible_activations
    }
    return config_as_dict

def load_config() -> dict:
    conf = get_config()
    conf_dict = unpack_config(conf)
    return conf_dict

# returns a list with all experiments names 
def experiments_names_splitter(experiments_dir: str)->list:
    experiments_names_entities = []
    for name in os.listdir(experiments_dir):
        entities = name.split('_')
        experiments_names_entities.append(entities)
    return experiments_names_entities

def as_hiddens_configs(entities: list):
    list_of_tups = []
    for lst in entities:
        hiddens = [ (int(lst[i]), lst[i+1]) for i in range(0, len(lst), 2) ]
        list_of_tups.append(hiddens)
    return list_of_tups


def prepare_experiments_list(conf: dict) -> list:
    neurons = conf['neurons_cardinality_per_layer']
    all_experiments_configs = []
    for activations in conf['possible_activations']:
        current_hidden_config = list(zip(neurons, activations))
        all_experiments_configs.append(current_hidden_config)
    return all_experiments_configs

def get_data(batch_size: int = 64):
    transform = T.Compose([T.Resize(70), T.CenterCrop((64,64)), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])])
    ds_train = GTSRB('../_data/', split='train', transform=transform, download=False)
    ds_test = GTSRB('../_data/', split='test', transform=transform, download=False)
    train_loader = DataLoader(ds_train, batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size, shuffle=False)
    if train_loader and test_loader:
        print("Data collected!")
        return ds_train, train_loader, ds_test, test_loader
    else:
        raise ValueError("Something went wrong: train_loader or test_loader not loaded as expected.")
    
def prepare_model_dirs(hiddens_layers_configs):
    dirs_list = []
    for act in hiddens_layers_configs:
        config_name = "_".join([f"{n}_{a}" for n, a in act])
        dirs_list.append(config_name)
    return dirs_list


def load_encoder(model_path: str, device, input_space_dim: int, hidden_layers_config: list, latent_space_dim: int):
    encoder = AutoEncoder(input_space_dim, hidden_layers_config, latent_space_dim)
    file_prefix = model_path.replace("/", "_").replace("\\", "_") 
    full_path = os.path.join("../experiments", model_path, f"{file_prefix}_last_model.pth")
    saved_weights = torch.load(full_path, map_location=device)
    encoder.load_state_dict(saved_weights)
    encoder.decoder = torch.nn.Identity()
    encoder.to(device)
    encoder.eval() 
    return encoder


def encode_inputs(model, data_loader, device, normalize: bool = True, save_path: str = None, motivation: str = None):
    embeddings_list = []
    labels_list = [] # keeps track of the class (label) of each embedding
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            embeddings = model(images) # asking the model to produce the embeddings
            for emb, lbl in zip(embeddings, labels):
                embeddings_list.append(emb.cpu()) # taking the embeddings of the current batch and appending to the list
                labels_list.append(lbl) # taking the classes of the current batch and appending them to the list
        # every 512_dimensional tensor will become [1, 512]. vstack, vertically stacks them on first dimension 
        embeddings = torch.vstack(embeddings_list) 
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        labels = torch.vstack(labels_list)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        embeddings_filepath = os.path.join(save_path, "embeddings.pkl")
        with open(embeddings_filepath, "wb") as file:
            pickle.dump({"embeddings": embeddings, "labels": labels}, file)
        if motivation is None:
            print("[WARNING] You are saving embeddings without mentioning any motivation!")
        else:
            motivation_txt_path = os.path.join(save_path, "readme.txt")
            with open(motivation_txt_path, "w") as file:
                file.write("Embeddings explaination: {}".format(motivation))
    return (embeddings, labels)
