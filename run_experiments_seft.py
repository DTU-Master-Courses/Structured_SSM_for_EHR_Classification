import os
import itertools
import subprocess
import hashlib

# Define the parameter grid
param_grid = {
    'epochs': [50, 100],
    'batch_size': [64, 128],
    'dropout': [0.3, 0.4],
    'attn_dropout': [0.2, 0.3],
    'heads': [2, 4],
    'lr': [0.001, 0.01],
    'seft_dot_prod_dim': [256, 512],
    'seft_n_phi_layers': [1, 2],
    'seft_n_psi_layers': [3, 5],
    'seft_n_rho_layers': [2, 3],
    'seft_phi_dropout': [0.2, 0.3],
    'seft_phi_width': [256, 512],
    'seft_psi_width': [16, 32],
    'seft_psi_latent_width': [64, 128],
    'seft_latent_width': [32, 64],
    'seft_rho_dropout': [0.0, 0.1],
    'seft_rho_width': [128, 256],
    'seft_max_timescales': [500, 1000],
    'seft_n_positional_dims': [8, 16]
}

# Generate all combinations of parameters
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Directory to save the results
output_dir = './SSSM/Structured_SSM_for_EHR_Classification/seft_output/'

# Run the model for each combination of parameters
for params in combinations:
    # Create a unique hash for the parameter combination
    param_str = '_'.join(f"{key}={value}" for key, value in params.items())
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    output_path = os.path.join(output_dir, param_hash)
    
    # Create the directory if it does not exist
    os.makedirs(output_path, exist_ok=True)
    
    command = (
        f"python cli.py --output_path='{output_path}' "
        f"--model_type=seft --epochs={params['epochs']} --batch_size={params['batch_size']} "
        f"--dropout={params['dropout']} --attn_dropout={params['attn_dropout']} "
        f"--heads={params['heads']} --lr={params['lr']} "
        f"--seft_dot_prod_dim={params['seft_dot_prod_dim']} --seft_n_phi_layers={params['seft_n_phi_layers']} "
        f"--seft_n_psi_layers={params['seft_n_psi_layers']} --seft_n_rho_layers={params['seft_n_rho_layers']} "
        f"--seft_phi_dropout={params['seft_phi_dropout']} --seft_phi_width={params['seft_phi_width']} "
        f"--seft_psi_width={params['seft_psi_width']} --seft_psi_latent_width={params['seft_psi_latent_width']} "
        f"--seft_latent_width={params['seft_latent_width']} --seft_rho_dropout={params['seft_rho_dropout']} "
        f"--seft_rho_width={params['seft_rho_width']} --seft_max_timescales={params['seft_max_timescales']} "
        f"--seft_n_positional_dims={params['seft_n_positional_dims']}"
    )
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

    # Optionally, you can parse the output and save the performance metrics
    # For example, you can read the summary.json file and extract the metrics