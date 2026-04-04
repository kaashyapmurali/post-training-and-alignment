from datetime import datetime
from zoneinfo import ZoneInfo
import json
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
import torch

class SFT:
    def __init__(self, config):
        self._load(config)

    def _load(self, config):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(config["model"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = load_dataset(config["dataset"])
        print(f"Loaded model={config['model']}, dataset={config['dataset']}")

    def fine_tune(self):
        config = self.config
        run_id = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
        output_dir = f"{config['ROOT']}/outputs/sft/{run_id}"
        if config['DEV']:
            self.dataset['train'] = self.dataset["train"].select(range(100))
        dt = self.dataset["train"].train_test_split(test_size=0.2)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=config["lora"]["target_modules"],
            inference_mode=False,
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
        )
        trainer_args = SFTConfig(
            learning_rate=config["training"]["learning_rate"],
            output_dir=output_dir,
            num_train_epochs=config["training"]["epochs"],
            max_steps=-1 if not config["DEV"] else 10,
            per_device_train_batch_size=config["training"]["train_batch_size"] if not config['DEV'] else 1,
            per_device_eval_batch_size=config["training"]["eval_batch_size"] if not config['DEV'] else 1,
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            logging_steps=config["training"]["logging_steps"],
            save_steps=config["training"]["save_steps"],
            eval_steps=config["training"]["eval_steps"] if not config["DEV"] else 5,
            eval_strategy="steps",
            bf16=torch.cuda.is_available() or torch.backends.mps.is_available(),
        )
        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=trainer_args,
            train_dataset=dt["train"],
            eval_dataset=dt["test"],
            formatting_func=format_alpaca,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.save_model(f"{output_dir}/final")
        with open(f"{output_dir}/log_history.json", "w") as f:
            json.dump(trainer.state.log_history, f)


def format_alpaca(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )

