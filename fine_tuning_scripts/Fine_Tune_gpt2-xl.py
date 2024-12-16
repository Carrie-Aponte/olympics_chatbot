# Fine tuning gpt2-xl on my Olympics dataset :)

import bitsandbytes as bnb
import torch
import pandas as pd

from transformers import GPT2Tokenizer, GPT2LMHeadModel, BitsAndBytesConfig, GenerationConfig, TFTrainingArguments
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "gpt2-xl"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto") # had to leave out the 4 bit quantization for now.. it was giving errors unfortunately. adding back in for beocat runs.

tokenizer.pad_token = tokenizer.eos_token

file_path = "/homes/cgaponte/ProgDS/TermProject/combined_qa_data.csv"
olympics_df = pd.read_csv(file_path)

train_df, val_df = train_test_split(olympics_df, test_size=0.2, random_state=42)

# getting it in the right format for hugging face as a dataset dict
def prepare_hf_dataset(df):
    return Dataset.from_pandas(df)

train_dataset = prepare_hf_dataset(train_df)
val_dataset = prepare_hf_dataset(val_df)

dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

def input_tokenizer(examples):
    context = examples["context"]
    question = examples["question"]
    answer = examples["answer"]
    
    input_prompt = f"{context}\n### Question: {question}\n### Response: {answer}"
    return {"text": input_prompt}

mapped_dataset = dataset.map(input_tokenizer, remove_columns=dataset["train"].column_names, load_from_cache_file=False) # mapping dataset to be the right format

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)



training_args = TrainingArguments(
    output_dir="/homes/cgaponte/ProgDS/TermProject/gpt2-xl_olympics_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    save_steps=100,
    save_total_limit=2,
    eval_steps=100,
    eval_strategy="epoch",
    report_to="none",
    fp16=True 
)

trainer = SFTTrainer(
    model=model,
    train_dataset=mapped_dataset["train"],
    eval_dataset=mapped_dataset["validation"],
    args=training_args,
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512
)

trainer.train()

trainer.save_model("/homes/cgaponte/ProgDS/TermProject/gpt2-xl_olympics_finetuned_model")

eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)
