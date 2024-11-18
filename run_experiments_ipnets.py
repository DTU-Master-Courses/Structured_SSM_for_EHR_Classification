import os
import itertools
import subprocess
import hashlib

# Define the parameter grid
param_grid = {
    'epochs': [50, 100],
    'batch_size': [16, 32],
    'lr': [0.0001, 0.001],
    'ipnets_imputation_stepsize': [1, 2],
    'ipnets_reconst_fraction': [0.5, 0.75],
    'recurrent_dropout': [0.2, 0.3],
    'recurrent_n_units': [32, 64]
}

# Generate all combinations of parameters
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Directory to save the results
output_dir = './SSSM/Structured_SSM_for_EHR_Classification/ipnets_output/'

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
        f"--model_type=ipnets --epochs={params['epochs']} --batch_size={params['batch_size']} "
        f"--lr={params['lr']} --ipnets_imputation_stepsize={params['ipnets_imputation_stepsize']} "
        f"--ipnets_reconst_fraction={params['ipnets_reconst_fraction']} --recurrent_dropout={params['recurrent_dropout']} "
        f"--recurrent_n_units={params['recurrent_n_units']}"
    )
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

    # Optionally, you can parse the output and save the performance metrics
    # For example, you can read the summary.json file and extract the metrics