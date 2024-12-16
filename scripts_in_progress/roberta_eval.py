# evaluating my roberta model!! lets gooooooooooooooooooo

import os
import torch
from datasets import load_from_disk
from transformers import RobertaForQuestionAnswering, RobertaTokenizerFast
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu.metrics import BLEU
from statistics import mean
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


model_path = "/homes/cgaponte/ProgDS/TermProject/roberta_olympics_finetuned"
dataset_path = "/homes/cgaponte/ProgDS/TermProject/mapped_bert_olympics_dataset"


tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
model = RobertaForQuestionAnswering.from_pretrained(model_path).to(device)


mapped_dataset = load_from_disk(dataset_path)
val_dataset = mapped_dataset["validation"]

def tokenize_dataset(example):
    tokenized = tokenizer(
        example["context"],
        example["question"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True,
    )
    tokenized["example_id"] = example["id"]
    tokenized["context"] = example["context"]
    tokenized["answer"] = example.get("answer", "N/A")
    return tokenized

val_dataset = val_dataset.map(tokenize_dataset, batched=True)


val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "offset_mapping", "context", "answer", "example_id", "question"]
)


dataloader = DataLoader(val_dataset, batch_size=4)

example_ids = [str(example["example_id"]) for example in val_dataset]
if len(set(example_ids)) != len(example_ids):
    raise ValueError("Duplicate example_id found in val_dataset.")


def postprocess_qa_predictions(examples, features, predictions):
    all_start_logits, all_end_logits = predictions

    example_id_to_index = {str(k): i for i, k in enumerate(examples["example_id"])}
    features_per_example = defaultdict(list)

    print("Example ID to Index Mapping (Sample):", list(example_id_to_index.items())[:5])


    for i, feature in enumerate(features):
        if "example_id" not in feature:
            print(f"Feature missing 'example_id': {feature}")
            raise KeyError("Feature missing 'example_id'")
        example_id = str(feature["example_id"])
        if example_id not in example_id_to_index:
            print(f"Warning: Example ID {example_id} not found in `example_id_to_index`.")
            continue 
        features_per_example[example_id_to_index[example_id]].append(i)

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
    print(f"Generating predictions for {len(val_dataset)} examples...")
    all_start_logits = []
    all_end_logits = []
    features = []

    for batch in dataloader:
        inputs = {key: batch[key].to(device) for key in batch if key in ["input_ids", "attention_mask"]}
        with torch.no_grad():
            outputs = model(**inputs)

        all_start_logits.extend(outputs.start_logits.cpu().numpy())
        all_end_logits.extend(outputs.end_logits.cpu().numpy())

        batch_features = [{key: batch[key][i] for key in batch} for i in range(len(batch["input_ids"]))]
        features.extend(batch_features)

    predictions = postprocess_qa_predictions(val_dataset, features, (all_start_logits, all_end_logits))
    return predictions



def evaluate(predictions):
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
write_predictions_to_file(predictions, f"{model_path}_predictions.txt")
