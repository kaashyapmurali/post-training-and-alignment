import yaml
from pathlib import Path
from train import SFT

config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
config['ROOT'] = Path.cwd()

sft = SFT(config=config)
sft.fine_tune()
