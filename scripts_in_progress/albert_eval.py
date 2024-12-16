# just evaluating my trained albert!!!! go albert go!!!

import os
import torch
import pandas as pd
import re
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AlbertForQuestionAnswering,
    AlbertTokenizerFast,
)
from torch.utils.data import DataLoader
from collections import defaultdict
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu.metrics import BLEU
from statistics import mean
from torch.utils.data.dataloader import default_collate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")



model_path = "/homes/cgaponte/ProgDS/TermProject/albert_olympics_finetuned" 
dataset_path = "/homes/cgaponte/ProgDS/TermProject/mapped_bert_olympics_dataset" 
model_name = "albert-base-v2"

tokenizer = AlbertTokenizerFast.from_pretrained(model_path)
try:
    model = AlbertForQuestionAnswering.from_pretrained(model_path).to(device)
except Exception as e:
    print(f"Error loading model: {e}. Reinitializing qa_outputs.")
    model = AlbertForQuestionAnswering.from_pretrained("albert-base-v2")
    model.qa_outputs = torch.nn.Linear(model.config.hidden_size, 2).to(device)



mapped_dataset = load_from_disk(dataset_path)
val_dataset = mapped_dataset["validation"]

required_keys = ["context", "question", "answer", "id"]
if not all(key in val_dataset.column_names for key in required_keys):
    raise ValueError(f"Missing required keys in validation dataset: {required_keys}")

def tokenize_dataset(example, idx):
    context = example["context"]
    question = example["question"]

    if isinstance(context, list) and isinstance(question, list):
        tokenized_batch = tokenizer(
            context,
            question,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_offsets_mapping=True,
        )
        
        tokenized_batch["example_id"] = [
            str(f"{hash(c + q)}_{i}") for i, (c, q) in enumerate(zip(context, question))
        ]
        tokenized_batch["context"] = context
        tokenized_batch["answer"] = example.get("answer", ["N/A"] * len(context))
    else:
        tokenized = tokenizer(
            context,
            question,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_offsets_mapping=True,
        )
        
        tokenized["example_id"] = str(f"{hash(context + question)}_{idx}")
        tokenized["context"] = context
        tokenized["answer"] = example.get("answer", "N/A")

        tokenized_batch = {key: [value] for key, value in tokenized.items()}

    return tokenized_batch



val_dataset = val_dataset.map(
    lambda example, idx: tokenize_dataset(example, idx=idx),
    batched=True,
    with_indices=True,
)

val_dataset = val_dataset.map(
    lambda example: {
        "example_id": example["example_id"] if isinstance(example["example_id"], str) else str(example["example_id"][0])
    },
    batched=False
)

example_ids = [example["example_id"] for example in val_dataset]
if not all(isinstance(eid, str) for eid in example_ids):
    raise ValueError("Some `example_id` values are not strings. Check your tokenization function.")




val_dataset = val_dataset.filter(lambda x: "example_id" in x and x["example_id"] is not None)

val_dataset = val_dataset.map(
    lambda example: {"example_id": example["example_id"] if isinstance(example["example_id"], str) else str(example["example_id"][0])},
    batched=False
)


val_dataset = val_dataset.filter(
    lambda x: x["context"] and x["question"], 
    batched=False
)

print("Post-tokenization dataset size:", len(val_dataset))

val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "offset_mapping", "context", "answer", "example_id", "question"],
)

def clean_text(text):
    if not isinstance(text, str):
        return text
    return re.sub(f"[\[\]]" "", text).strip()

def clean_example(example):
    example["context"] = clean_text(example["context"])
    example["question"] = clean_text(example["question"])
    example["answer"] = clean_text(example["answer"])
    return example

def preprocess_dataset(dataset):
    return dataset.map(clean_example, batched=False)

val_dataset = preprocess_dataset(val_dataset)

print("Checking dataset structure...")
print("Sample val_dataset entry:", val_dataset[0])

example_ids = [example["example_id"] for example in val_dataset]
if len(set(example_ids)) != len(example_ids):
    raise ValueError("Duplicate example_id found in val_dataset.")


dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=default_collate)

for idx, batch in enumerate(dataloader):
    if "example_id" not in batch:
        print(f"Batch {idx} missing `example_id`. Skipping...")
        continue

example_id_to_index = {example["example_id"]: idx for idx, example in enumerate(val_dataset)}

missing_ids = [example["example_id"] for example in val_dataset if example["example_id"] not in example_id_to_index]
if missing_ids:
    print(f"Missing example IDs: {missing_ids}")


features = []

for batch in dataloader:
    features.append(batch)

example_id_to_index = {example["example_id"]: idx for idx, example in enumerate(val_dataset)}

for idx, example in enumerate(val_dataset):
    if "example_id" not in example:
        print(f"Missing `example_id` in dataset entry {idx}: {example}")


for idx, feature in enumerate(features):
    if "example_id" not in feature:
        print(f"Missing `example_id` in feature {idx}: {feature}")
        continue
    if feature["example_id"] not in example_id_to_index:
        print(f"Warning: example_id {feature['example_id']} not found in example_id_to_index.")



def postprocess_qa_predictions(examples, features, predictions):
    all_start_logits, all_end_logits = predictions
    example_id_to_index = {str(k): i for i, k in enumerate(examples["example_id"])}
    features_per_example = defaultdict(list)

    for idx, feature in enumerate(features):
        if "example_id" not in feature:
            print(f"Missing `example_id` in feature {idx}: {feature}")
            continue
        if isinstance(feature["example_id"], list):
            feature["example_id"] = str(feature["example_id"][0]) 
        if feature["example_id"] not in example_id_to_index:
            print(f"Warning: example_id {feature['example_id']} not found in example_id_to_index.")
            continue
        features_per_example[example_id_to_index[feature["example_id"]]].append(idx)


    predictions = {}
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        if not feature_indices:
            print(f"Warning: No features found for example_id {example['example_id']}. Skipping...")
            predictions[example["example_id"]] = ""
            continue

        min_null_score = None
        valid_answers = []

        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            cls_index = (features[feature_index]["input_ids"] == tokenizer.cls_token_id).nonzero(as_tuple=True)[0].item()
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            for start_index, start_logit in enumerate(start_logits):
                for end_index, end_logit in enumerate(end_logits[start_index:]):
                    end_index += start_index
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                        "score": start_logit + end_logit,
                        "text": context[start_char:end_char],
                    })

        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x["score"])
        else:
            best_answer = {"text": "", "score": min_null_score}

        predictions[example["example_id"]] = best_answer["text"]
    return predictions



def generate_predictions():
    print(f"Generating predictions for {len(val_dataset)} examples for {model_name}...")
    all_start_logits = []
    all_end_logits = []
    features = []

    for batch in dataloader:
        if not isinstance(batch, dict) or "example_id" not in batch:
            print(f"Invalid or missing `example_id` in batch {idx}: {batch}")
            continue
        if not batch:
            print(f"Empty batch at index {idx}, skipping...")
            continue
        
        inputs = {key: batch[key].to(device) for key in batch if key in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            outputs = model(**inputs)

        all_start_logits.append(outputs.start_logits.cpu().numpy())
        all_end_logits.append(outputs.end_logits.cpu().numpy())
        features.append(batch)

    predictions = postprocess_qa_predictions(val_dataset, features, (all_start_logits, all_end_logits))
    return predictions



def evaluate(predictions):
    print("Evaluating predictions...")
    pred_texts = list(predictions.values())
    true_texts = [example["answer"] for example in val_dataset]

    bleu = BLEU()
    bleu_score = bleu.corpus_score(pred_texts, [true_texts]).score

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, true in zip(pred_texts, true_texts):
        scores = rouge.score(true, pred)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    avg_rouge = {key: mean(values) for key, values in rouge_scores.items()}

    _, _, bert_scores = bert_score(pred_texts, true_texts, lang="en", verbose=True)
    avg_bert_score = bert_scores.mean().item()

    print(f"BLEU Score: {bleu_score:.4f}")
    for key, value in avg_rouge.items():
        print(f"{key.upper()} Score: {value:.4f}")
    print(f"BERTScore: {avg_bert_score:.4f}")

    
    eval_output_file = f"{model_path}_evaluation_metrics.txt"
    with open(eval_output_file, "w") as f:
        f.write(f"BLEU Score: {bleu_score:.4f}\n")
        for key, value in avg_rouge.items():
            f.write(f"Average {key.upper()} Score: {value:.4f}\n")
        f.write(f"Average BERTScore: {avg_bert_score:.4f}\n")
    print(f"Metrics saved to {eval_output_file}")



def write_predictions_to_file(predictions, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, example in enumerate(val_dataset):
            example_id = example["example_id"]
            generated_answer = predictions.get(example_id, "No prediction available")
            real_answer = example["answer"]

            f.write(f"Example {idx + 1}:\n")
            f.write(f"Question: {example['question']}\n")
            f.write(f"Context: {example['context'][:500]}...\n")
            f.write(f"Generated Answer: {generated_answer}\n")
            f.write(f"Real Answer: {real_answer}\n")
            f.write("-" * 50 + "\n")

    print(f"Predictions written to {output_file}")


predictions = generate_predictions()
evaluate(predictions)
write_predictions_to_file(predictions, "alberto_generated_vs_real_answers.txt")
