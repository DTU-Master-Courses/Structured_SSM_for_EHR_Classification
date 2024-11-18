import os
import itertools
import subprocess

# Define the parameter grid
param_grid = {
    'epochs': [50, 100],
    'batch_size': [16, 32],
    'dropout': [0.2, 0.3],
    'attn_dropout': [0.1, 0.2],
    'layers': [3, 4],
    'heads': [1, 2],
    'pooling': ['max', 'mean'],
    'lr': [0.0001, 0.001]
}

# Generate all combinations of parameters
keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Directory to save the results
output_dir = './SSSM/Structured_SSM_for_EHR_Classification/output/'

# Run the model for each combination of parameters
for params in combinations:
    output_path = os.path.join(output_dir, f"epochs={params['epochs']}_batch_size={params['batch_size']}_dropout={params['dropout']}_attn_dropout={params['attn_dropout']}_layers={params['layers']}_heads={params['heads']}_pooling={params['pooling']}_lr={params['lr']}")
    command = (
        f"python cli.py --output_path='{output_path}' "
        f"--epochs={params['epochs']} --batch_size={params['batch_size']} "
        f"--model_type=transformer --dropout={params['dropout']} "
        f"--attn_dropout={params['attn_dropout']} --layers={params['layers']} "
        f"--heads={params['heads']} --pooling={params['pooling']} "
        f"--lr={params['lr']}"
    )
    print(f"Running: {command}")
    subprocess.run(command, shell=True)

    # Optionally, you can parse the output and save the performance metrics
    # For example, you can read the summary.json file and extract the metrics