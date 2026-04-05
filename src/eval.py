import lm_eval
from lm_eval.models.huggingface import HFLM
from peft import PeftModel
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import json
import os

class Evaluator:
    def __init__(self, config):
        self._load(config)

    def _load(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.output_dir = f"{config['ROOT']}/outputs/sft/{config['run_id']}"
        print(f"Loaded tokenizer={config['model']}")

    def _get_model(self, type="base"):
        assert type in ["base", "finetune"], "wrong model type specified"

        if type == "base":
            model = AutoModelForCausalLM.from_pretrained(self.config["model"], device_map="auto")
            print(f"Running eval on base model, model={self.config['model']}")
            return model

        elif type == "finetune":
            config = self.config
            run_id = config['run_id']
            eval_run_id = run_id if config['finetune'] else config['eval-runid']
            assert eval_run_id is not None, "eval_run_id can't be None"
            print(f"Running eval on fine-tuned model, model={config['model']}, eval_run_id={eval_run_id}")
            adapter_dir = f"{config['ROOT']}/outputs/sft/{eval_run_id}"
            base_model = AutoModelForCausalLM.from_pretrained(self.config["model"], device_map="auto")
            model = PeftModel.from_pretrained(base_model, f"{adapter_dir}/final")
            return model

    def run_eval(self, type="base"):
        model = self._get_model(type=type)

        lm = HFLM(
            pretrained=model,
            tokenizer=self.tokenizer
        )
        tasks_list = ["hellaswag", "mmlu"]
        print(f"Eval tasks={tasks_list}")        
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks_list,
            batch_size="auto" if not self.config['DEV'] else 1,
            limit=None if not self.config['DEV'] else 2
        )
        print("Eval completed")
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f"{self.output_dir}/eval_{type}.json", "w") as f:
            json.dump(results["results"], f)
        return results