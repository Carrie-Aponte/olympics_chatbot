# Training the gpt2-large (trained on SQuAD) on my Olympics dataset :)

import bitsandbytes as bnb
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BitsAndBytesConfig, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict
from peft import get_peft_model, LoraConfig
from sklearn.model_selection import train_test_split

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

peft_model_id = "/homes/cgaponte/DeepLearning/Term_Project/qlora_large_gpt2_fulldataset"  # using the gpt2-large model that I trained on the entire SQuAD dataset
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = GPT2LMHeadModel.from_pretrained(peft_model_id)

file_path = "/homes/cgaponte/ProgDS/TermProject/combined_qa_data.csv"
olympics_df = pd.read_csv(file_path)

train_df, val_df = train_test_split(olympics_df, test_size=0.2, random_state=42)

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

mapped_dataset = dataset.map(
    input_tokenizer,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


def preprocess_dataset(examples):
    inputs = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs

tokenized_dataset = mapped_dataset.map(
    preprocess_dataset,
    batched=True,
    remove_columns=["text"],
    load_from_cache_file=False
)

training_args = TrainingArguments(
    output_dir="./squad-gpt2-large-olympics-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=1,
    save_steps=100,
    save_total_limit=2,
    eval_steps=100,
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    report_to="none",
    fp16=True 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer
)

trainer.train()

model_save_path = "./squad-gpt2-large-olympics-finetuned"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"Model saved to {model_save_path}")

def analyse_zero_shot_model(data, idx, tokenizer, model):
    prompt = data[idx]["context"]
    question = data[idx]["question"]
    answer = data[idx]["answer"]

    print("Context:")
    print(prompt)
    print("\nQuestion:")
    print(question)
    print("\nExpected Answer:")
    print(answer)

    input_text = f"{prompt}\n### Question: {question}\n### Response:"
    tokenized_input = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    device = model.device
    output_ids = model.generate(
        input_ids=tokenized_input["input_ids"].to(device),
        attention_mask=tokenized_input["attention_mask"].to(device),
        max_new_tokens=128,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        num_beams=5,
        early_stopping=True
    )

    predicted_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nGenerated Answer:")
    print(predicted_output)


index = 0
analyse_zero_shot_model(mapped_dataset["validation"], index, tokenizer, model)