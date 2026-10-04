import torch
import yaml 
from torchvision.datasets import GTSRB
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from utils import Experiment, AutoEncoder, set_deterministic_seed
from itertools import product
import copy
import time
import os


def get_config():
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    print("Loaded configuration file produced: {}".format(conf_dict))
    return conf_dict

def get_data(batch_size: int = 64):
    transform = T.Compose([T.Resize(70), T.CenterCrop((64,64)), T.ToTensor(), T.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229,0.224, 0.225])])
    ds_train = GTSRB('../_data/', split='train', transform=transform, download=True)
    ds_test = GTSRB('../_data/', split='test', transform=transform, download=True)
    train_loader = DataLoader(ds_train, batch_size, shuffle=False)
    test_loader = DataLoader(ds_test, batch_size, shuffle=False)
    if train_loader and test_loader:
        print("Data collected!")
        return ds_train, train_loader, ds_test, test_loader
    else:
        raise ValueError("Something went wrong: train_loader or test_loader not loaded as expected.")

# For all possibile activations function attaches the relative one to the layer with cardinality of neurons
def prepare_experiments_list(conf: dict) -> list:
    neurons = conf['neurons_cardinality_per_layer']
    all_experiments_configs = []
    for activations in conf['possible_activations']:
        current_hidden_config = list(zip(neurons, activations))
        all_experiments_configs.append(current_hidden_config)
    return all_experiments_configs

def main():
    EXPERIMENTS_SEEDS = [42, 999, 2026]
    configuration_dictionary = load_config()
    ds_train, train_loader, ds_test, test_loader = get_data(batch_size=configuration_dictionary['bs'])
    all_experiments_configs = prepare_experiments_list(configuration_dictionary)
    print("Available experiments: {}".format(all_experiments_configs))
    
    for act in all_experiments_configs:
        config_name = "_".join([f"{n}_{a}" for n, a in act])
        print("\n=======================================================")
        print(" === Experiment on configuration: {} ===".format(act))
        print("=======================================================")
        
        # Running each config on each seed
        for seed in EXPERIMENTS_SEEDS:
            set_deterministic_seed(seed)
            print("\n---> Starting run for {} with SEED: {}".format(config_name, seed))
            ae_model = AutoEncoder(configuration_dictionary['input_space_dim'], act, configuration_dictionary['latent_space_dim'])
            if not ae_model:
                raise ValueError("Something went wrong while creating the model.")
            current_run_title = "{}_seed_{}".format(config_name, seed)
            current_run_dir = os.path.join(config_name, "seed_{}".format(seed))
            current_training_data = copy.deepcopy(ds_train)
            exp = Experiment(training_data=current_training_data, testing_data=ds_test, title=current_run_title,
                motivation="Evaluating linear AE embeddings over {} (Seed {})".format(act, seed),
                directory_name=current_run_dir, need_valset=True, valset_proportion=0.2)

            ae_model.to(exp.device)
            optimizer = torch.optim.AdamW(ae_model.parameters(), lr=float(configuration_dictionary['lr']),weight_decay=float(configuration_dictionary['wd']))

            exp.start_experiment(model=ae_model, num_epochs=configuration_dictionary['num_epochs'], 
                in_batch_size=configuration_dictionary['bs'], learning_rate=configuration_dictionary['lr'], optim=optimizer,
                verbose=1)

            print("=== Run SEED {} finished. Mem Cleaning. ===".format(seed))
            del ae_model
            del optimizer
            del exp
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            time.sleep(3) 
            
        print("\n=== Configuration {} completed for all seeds. 10 seconds pause... ===".format(config_name))
        time.sleep(10)


if __name__ == "__main__":
    main()