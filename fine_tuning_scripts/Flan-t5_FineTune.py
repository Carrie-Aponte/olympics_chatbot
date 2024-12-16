# Fine tuning flan-t5 on my Olympics dataset :)

import bitsandbytes as bnb
import torch
import pandas as pd

from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score
import numpy as np
import nltk


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "google/flan-t5-large"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy = True)

model = T5ForConditionalGeneration.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
model.gradient_checkpointing_enable()
model.config.use_cache = False

tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,  
    lora_alpha=32,  
    lora_dropout=0.1,  
    task_type=TaskType.SEQ_2_SEQ_LM  
)

model = get_peft_model(model, lora_config)

for param in model.parameters():
    if param.dtype in [torch.float16, torch.float32, torch.bfloat16]:
        param.data = param.data.to(torch.float32)
        param.requires_grad = True

model.train()


file_path = "/homes/cgaponte/ProgDS/TermProject/combined_qa_data.csv"
olympics_df = pd.read_csv(file_path)

olympics_df["answer"] = olympics_df["answer"].fillna("").astype(str) # just in case there are null values - i'm getting weird errors.

train_df, val_df = train_test_split(olympics_df, test_size=0.2, random_state=42)

# getting it in the right format for hugging face as a dataset dict
def prepare_hf_dataset(df):
    return Dataset.from_pandas(df)

train_dataset = prepare_hf_dataset(train_df)
val_dataset = prepare_hf_dataset(val_df)

dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

def preprocess_function(examples):
    inputs = [f"Question: {question} Context: {context}" for question, context in zip(examples["question"], examples["context"])]
    targets = examples["answer"]
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)["input_ids"]
    
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    model_inputs["labels"] = labels
    
    return model_inputs


train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)



training_args = TrainingArguments(
    output_dir="/homes/cgaponte/ProgDS/TermProject/flan-t5-olympics-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    save_steps=100,
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    report_to="none",
    fp16=False # trying false now 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()


eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)


trainer.save_model("/homes/cgaponte/ProgDS/TermProject/flan-t5-olympics-finetuned")
tokenizer.save_pretrained("/homes/cgaponte/ProgDS/TermProject/flan-t5-olympics-finetuned-tokenizer")


sample = val_dataset[0]
input_text = f"Question: {sample['question']} Context: {sample['context']}"

input_encodings = tokenizer(
    input_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)

input_ids = input_encodings["input_ids"].to(model.device)
attention_mask = input_encodings["attention_mask"].to(model.device)

outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=128,
    temperature=0.3,
    top_p=0.9,
    num_beams=5,
    early_stopping=True
)

def postprocess_text(text):
    text = text.strip()
    text = " ".join(text.split()) 
    return text

def save_generated_vs_actual(predictions, actuals, output_file="flan_t5_generated_vs_actual.txt"):
    """Save generated vs. actual answers to a file for manual inspection."""
    with open(output_file, "w") as f:
        for i, (gen, act) in enumerate(zip(predictions, actuals)):
            f.write(f"Example {i + 1}:\n")
            f.write(f"Generated: {gen}\n")
            f.write(f"Actual: {act}\n")
            f.write("-" * 50 + "\n")
    print(f"Generated vs. Actual answers saved to {output_file}")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=data_collator)

predictions = []
actual_answers = []

for batch in val_dataloader:
    inputs = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        attention_mask=attention_mask,
        max_new_tokens=128,
        temperature=0.3,
        top_p=0.9,
        num_beams=5,
        early_stopping=True
    )
    batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch_predictions = [postprocess_text(pred) for pred in batch_predictions] 
    predictions.extend(batch_predictions)

    batch_actuals = [postprocess_text(tokenizer.decode(ans, skip_special_tokens=True)) for ans in batch["labels"]]
    actual_answers.extend(batch_actuals)

save_generated_vs_actual(predictions, actual_answers)

with open("flan-t5_validation_predictions.txt", "w") as f:
    f.write("\n".join(predictions))


def calculate_metrics(predictions, actual_answers, metrics_file="flan_t5_evaluation_metrics.txt"):
    
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [actual_answers]).score

    rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for gen, ref in zip(predictions, actual_answers):
        scores = rouge_scorer_instance.score(ref, gen)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)
    avg_rouge_scores = {key: np.mean(value) for key, value in rouge_scores.items()}

    _, _, bert_scores = score(predictions, actual_answers, lang="en", verbose=True)
    avg_bert_score = bert_scores.mean().item()


    with open(metrics_file, "w") as f:
        f.write(f"BLEU Score: {bleu_score:.4f}\n")
        f.write("ROUGE Scores:\n")
        for metric, value in avg_rouge_scores.items():
            f.write(f"  {metric.upper()}: {value:.4f}\n")
        f.write(f"BERTScore: {avg_bert_score:.4f}\n")

    print(f"Metrics saved to {metrics_file}")

calculate_metrics(predictions, actual_answers)
