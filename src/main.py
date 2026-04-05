import yaml
import torch
from datetime import datetime, timezone
from pathlib import Path
from train import SFT
from eval import Evaluator

def printDevice():
    if torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        print("Device: MPS (Apple Silicon)")
    else:
        print("Device: CPU")

def getConfig():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['ROOT'] = Path(__file__).parent.parent
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config['run_id'] = run_id
    return config   

def run(config):
    # Training
    if config['finetune']:
        sft = SFT(config=config)
        sft.fine_tune()
    
    # Evaluation
    if config['eval-base'] or config['eval-finetune']:
        eval = Evaluator(config=config)
    
        # Base Model
        if config['eval-base']:
            results = eval.run_eval(type='base')
            print(results['results'])

        # Fine-tuned model
        if config['eval-finetune']:
            results = eval.run_eval(type='finetune')
            print(results['results'])

if __name__ == "__main__":
    printDevice()
    config = getConfig()
    run(config)