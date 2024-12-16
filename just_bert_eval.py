# evaluating my bert model! go bertttttt goooooooooooo

import os
import torch
from datasets import load_from_disk
from transformers import BertForQuestionAnswering, BertTokenizerFast
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu.metrics import BLEU
from statistics import mean
from collections import defaultdict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


model_path = "/homes/cgaponte/ProgDS/TermProject/bert_olympics_finetuned"
dataset_path = "/homes/cgaponte/ProgDS/TermProject/mapped_bert_olympics_dataset"

model_name = "bert-base-uncased"

tokenizer = BertTokenizerFast.from_pretrained(model_path)
try:
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)
except Exception as e:
    print(f"Error loading model: {e}. Reinitializing qa_outputs.")
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    model.qa_outputs = torch.nn.Linear(model.config.hidden_size, 2).to(device)


mapped_dataset = load_from_disk(dataset_path)
val_dataset = mapped_dataset["validation"]

print(f"Available keys in val_dataset: {val_dataset.column_names}")

def tokenize_dataset(example):
    tokenized = tokenizer(
        example["context"],
        example["question"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True,
    )

    if isinstance(example["context"], list) and isinstance(example["question"], list):
        tokenized["example_id"] = [
            str(hash(context + question)) for context, question in zip(example["context"], example["question"])
        ]
    else:
        tokenized["example_id"] = str(hash(example["context"] + example["question"]))

    return tokenized


val_dataset = val_dataset.map(tokenize_dataset, batched=True)

val_dataset = val_dataset.map(
    lambda example: {
        "example_id": example["example_id"] if isinstance(example["example_id"], str) else str(example["example_id"][0])
    },
    batched=False
)


val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "offset_mapping", "context", "answer", "example_id", "question"])

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

required_keys = ["input_ids", "attention_mask", "offset_mapping", "context", "question", "answer"]
if not all(key in val_dataset.column_names for key in required_keys):
    raise ValueError(f"Validation dataset is missing required keys: {required_keys}")


print(f"Sample tokenized dataset entry: {val_dataset[0]}")


dataloader = DataLoader(val_dataset, batch_size=4)

val_dataset = val_dataset.remove_columns(["__index_level_0__", "text"])

def align_predictions_with_references(predictions, val_dataset):
    aligned_pred_texts = []
    aligned_true_texts = []
    for example in val_dataset:
        example_id = example["example_id"]
        if example_id in predictions:
            aligned_pred_texts.append(predictions[example_id])
            aligned_true_texts.append(example["answer"])
        else:
            aligned_pred_texts.append("") 
            aligned_true_texts.append(example["answer"])
    return aligned_pred_texts, aligned_true_texts


def postprocess_qa_predictions(examples, features, predictions):
    all_start_logits, all_end_logits = predictions
    example_id_to_index = {example["example_id"]: i for i, example in enumerate(examples)}
    features_per_example = defaultdict(list)

    for i, feature in enumerate(features):
        example_id = feature["example_id"]
        if not isinstance(example_id, str):
            print(f"Invalid example_id at feature index {i}: {example_id}")
            continue
        if example_id in example_id_to_index:
            features_per_example[example_id_to_index[example_id]].append(i)
        else:
            print(f"Warning: Example ID {example_id} not found in example_id_to_index.")

    predictions = {}
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        min_null_score = None
        valid_answers = []

        context = example["context"]

        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]

            input_ids = features[feature_index]["input_ids"]
            if input_ids.ndim > 1:
                input_ids = input_ids[0]

            cls_index = (input_ids == tokenizer.cls_token_id).nonzero(as_tuple=True)[0].item()
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
                    
                    if not (len(offset_mapping[start_index]) == 2 and len(offset_mapping[end_index]) == 2):
                        print(f"Invalid offset_mapping entry: {offset_mapping[start_index]}")
                        continue


                    start_char, end_char = offset_mapping[start_index]
                    if not isinstance(start_char, int) or not isinstance(end_char, int):
                        print(f"Invalid offset mapping at index {start_index}: {offset_mapping[start_index]}")
                        continue

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
        if "input_ids" not in batch or "attention_mask" not in batch:
            print(f"Skipping batch due to missing keys. batch keys: {batch.keys()}")
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
    print("evaluating predictions!.............")
    pred_texts = list(predictions.values())
    true_texts = [example["answer"] for example in val_dataset if example["example_id"] in predictions]
    pred_texts, true_texts = align_predictions_with_references(predictions, val_dataset)


    print(f"Number of predictions: {len(pred_texts)}")
    print(f"Number of references: {len(true_texts)}")


    if len(pred_texts) != len(true_texts):
        print("Mismatch between predictions and references. Adjusting alignment.")
        aligned_pred_texts = []
        aligned_true_texts = []
        for example in val_dataset:
            example_id = example["example_id"]
            if example_id in predictions:
                aligned_pred_texts.append(predictions[example_id])
                aligned_true_texts.append(example["answer"])
        pred_texts = aligned_pred_texts
        true_texts = aligned_true_texts

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

predictions = {
    example["example_id"]: predictions.get(example["example_id"], "No Answer")
    for example in val_dataset
}


missing_predictions = [example["example_id"] for example in val_dataset if example["example_id"] not in predictions]
if missing_predictions:
    print(f"Missing predictions for {len(missing_predictions)} entries.")
    print(f"Example missing IDs: {missing_predictions[:5]}") 
else:
    print("All validation examples have corresponding predictions.")

evaluate(predictions)
write_predictions_to_file(predictions, f"{model_path}_predictions.txt")