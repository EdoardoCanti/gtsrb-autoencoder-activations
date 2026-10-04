import os
import torch
from utils import load_config
from utils import prepare_experiments_list, prepare_model_dirs, load_encoder, get_data, encode_inputs

def extract_config(config: dict):
    in_space_dim = config['input_space_dim']
    latent_dim = config['latent_space_dim']
    bs = config['bs']
    return in_space_dim, latent_dim, bs

def main():
    print("> Running: extract_embeddings")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    conf = load_config()
    INPUT_SPACE_DIM, LATENT_DIM, BATCH_SIZE = extract_config(conf)
    
    hiddens_layers_configs = prepare_experiments_list(conf)
    if hiddens_layers_configs:
        print("> Hidden layer configurations retrieved!\n")
    
    models_directories = prepare_model_dirs(hiddens_layers_configs)
    _, train_loader, _, test_loader = get_data(BATCH_SIZE)

    # External iteration for config
    for i, config_dir in enumerate(models_directories):
        print("\n======================================")
        print("Started encoding for configuration: {}".format(config_dir))
        print("======================================")
        base_path = os.path.join("../experiments", config_dir)
        if not os.path.exists(base_path):
            print(f"[WARNING] Directory not found: {base_path}. Skipping.")
            continue
        seed_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("seed_")]
        if not seed_dirs:
             print("[WARNING] No seed directories found in {}. Are you sure you ran the training?".format(base_path))
             continue

        for seed_dir in seed_dirs:
            print("\n> Processing {}...".format(seed_dir))
            current_model_path = os.path.join(config_dir, seed_dir)
            try:
                encoder = load_encoder(current_model_path, device, INPUT_SPACE_DIM, hiddens_layers_configs[i], LATENT_DIM)
                train_save_path = os.path.join(base_path, seed_dir, "training_embeddings")
                test_save_path = os.path.join(base_path, seed_dir, "test_embeddings")
                print("> Encoding training set")
                encode_inputs(encoder, train_loader, device, normalize=True, save_path=train_save_path)
                print("> Encoding test set ")
                encode_inputs(encoder, test_loader, device, normalize=True, save_path=test_save_path)
            except FileNotFoundError as e:
                print("[ERROR] Could not load model for {}. Error: {}".format(current_model_path, e))
        print("Finished encoding for configuration: {}".format(config_dir))


if __name__ == "__main__":
    main()