from pathlib import Path

def get_config():
    return {
        'batch_size': 16,  # Increased from 8 to 16 for better multi-GPU utilization
        "num_epochs": 15,
        "lr": 0.001,
        "seq_len": 128,
        "d_model": 512,
        "lang_src": "fi",
        "lang_tgt": "en",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        # Positional encoding type: "vanilla", "rope", or "relative_bias"
        "positional_encoding": "rope",  # Options: "vanilla", "rope", "relative_bias"
        # Legacy config for backward compatibility
        "use_rope": True,  # This will be ignored if positional_encoding is set
        # Relative bias specific settings
        "relative_bias_clamp_distance": 32  # Maximum relative distance for bias
    }
    
def get_weights_file_path(config, epoch:str):
    model_folder = Path(config['model_folder'])
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return model_folder / model_filename