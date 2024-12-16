# evaluating flan-t5 for olympics!!!! never thought the day would come :) :) :)

import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from bert_score import score
import numpy as np
import pandas as pd


model_path = "/homes/cgaponte/ProgDS/TermProject/flan-t5-olympics-finetuned"
tokenizer_path = "/homes/cgaponte/ProgDS/TermProject/flan-t5-olympics-finetuned-tokenizer"

model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, legacy=True)

file_path = "/homes/cgaponte/ProgDS/TermProject/combined_qa_data.csv"
olympics_df = Dataset.from_pandas(pd.read_csv(file_path))
_, val_dataset = olympics_df.train_test_split(test_size=0.2, seed=42).values()

val_dataset = val_dataset.filter(
    lambda x: isinstance(x["context"], str) and isinstance(x["question"], str) and isinstance(x["answer"], str)
)

def preprocess_function(examples):
    inputs = [
        f"Question: {question} Context: {context}" 
        for question, context in zip(examples["question"], examples["context"])
    ]
    targets = [answer if answer else "" for answer in examples["answer"]]
    
    if len(inputs) > 0:
        print(f"Sample Inputs: {inputs[:2]}")
        print(f"Sample Targets: {targets[:2]}")

    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512, return_tensors="pt")["input_ids"]

    labels = labels.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels

    return model_inputs


val_dataset = val_dataset.filter(lambda x: x["context"] and x["question"] and x["answer"] is not None)

for idx, example in enumerate(val_dataset):
    if not example["context"] or not example["question"] or not isinstance(example["answer"], str):
        print(f"Malformed example at index {idx}: {example}")


val_dataset = val_dataset.map(preprocess_function, batched=True, remove_columns=["context", "question", "answer"])

for idx, example in enumerate(val_dataset):
    if "labels" not in example:
        example["labels"] = torch.tensor(-1) 
    if isinstance(example["labels"], list):
        example["labels"] = torch.tensor(example["labels"])
    elif not isinstance(example["labels"], torch.Tensor):
        example["labels"] = torch.tensor(example["labels"]) 

for idx, example in enumerate(val_dataset):
    if "labels" not in example:
        print(f"Example at index {idx} is missing 'labels': {example}")
    elif not isinstance(example["labels"], torch.Tensor):
        print(f"Invalid 'labels' type at index {idx}: {type(example['labels'])}")
    elif not ((example["labels"] >= -100) & (example["labels"] < tokenizer.vocab_size)).all():
        print(f"Out-of-range token ID in labels at index {idx}: {example['labels']}")


data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
val_dataloader = DataLoader(val_dataset, batch_size=8, collate_fn=data_collator)

def postprocess_text(text):
    return " ".join(text.replace("<pad>", "").strip().split())

def save_generated_vs_actual(predictions, actuals, output_file="flan_t5_generated_vs_actual.txt"):
    with open(output_file, "w") as f:
        for i, (gen, act) in enumerate(zip(predictions, actuals)):
            if not gen.strip() or not act.strip():
                continue
            f.write(f"Example {i + 1}:\n")
            f.write(f"Generated: {gen}\n")
            f.write(f"Actual: {act}\n")
            f.write("-" * 50 + "\n")
    print(f"Generated vs. Actual answers saved to {output_file}")

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
        early_stopping=True,
        do_sample=True,
    )
    batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch_predictions = [postprocess_text(pred) for pred in batch_predictions]
    predictions.extend(batch_predictions)

    labels = batch["labels"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    batch_actuals = [
        postprocess_text(tokenizer.decode(ans[ans != -100], skip_special_tokens=True)) 
        for ans in labels
    ]
    
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
