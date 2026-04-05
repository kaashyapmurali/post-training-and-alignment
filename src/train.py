
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
        self.model = AutoModelForCausalLM.from_pretrained(config["model"], device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset = load_dataset(config["dataset"])
        print(f"Loaded model={config['model']}, dataset={config['dataset']}")
        print(f"Model device: {next(self.model.parameters()).device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    def fine_tune(self):
        config = self.config
        print(f"Fine-tuning model, model={config['model']}")
        output_dir = f"{config['ROOT']}/outputs/sft/{config['run_id']}"
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
            max_length=config["training"]["max_seq_length"],
            per_device_train_batch_size=config["training"]["train_batch_size"] if not config['DEV'] else 1,
            per_device_eval_batch_size=config["training"]["eval_batch_size"] if not config['DEV'] else 1,
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            logging_steps=config["training"]["logging_steps"],
            save_steps=config["training"]["save_steps"],
            eval_steps=config["training"]["eval_steps"] if not config["DEV"] else 5,
            eval_strategy="steps",
            bf16=torch.cuda.is_available() or torch.backends.mps.is_available(),
            packing=True
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
        print("Training completed")

def format_alpaca(example):
    return (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )