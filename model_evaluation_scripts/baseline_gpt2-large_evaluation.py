# Evaluating untrained gpt2-large on my Olympics dataset for a baseline

import torch
import pandas as pd
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset, Dataset, DatasetDict
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split

model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, device_map="auto")
tokenizer.pad_token = tokenizer.eos_token

file_path = "/homes/cgaponte/ProgDS/TermProject/combined_qa_data.csv"
olympics_df = pd.read_csv(file_path)

_, val_df = train_test_split(olympics_df, test_size=0.2, random_state=42) # not training so don't need that saved

def prepare_hf_dataset(df):
    return Dataset.from_pandas(df)

val_dataset = prepare_hf_dataset(val_df)

def preprocess_function(examples):
    context = examples["context"]
    question = examples["question"]
    answer = examples["answer"]
    return {
        "input_text": f"{context}\n### Question: {question}\n### Response:",
        "reference": answer
    }

val_dataset = val_dataset.map(preprocess_function, remove_columns=list(val_df.columns))


def generate_predictions(dataset, model, tokenizer, max_length=512):
    generated_answers = []
    reference_answers = []
    for example in tqdm(dataset, desc="Generating predictions"):
        input_text = example["input_text"]
        reference = example["reference"]

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(model.device)

        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            num_beams=5,
            early_stopping=True,
        )
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        generated_answers.append(generated_text.strip())
        reference_answers.append(reference.strip())

    return generated_answers, reference_answers

generated_answers, reference_answers = generate_predictions(val_dataset, model, tokenizer)


def evaluate_metrics(generated, reference):
    bleu = BLEU()
    bleu_score = bleu.corpus_score(generated, [reference]).score

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for gen, ref in zip(generated, reference):
        scores = rouge.score(ref, gen)
        rouge_scores["rouge1"].append(scores["rouge1"].fmeasure)
        rouge_scores["rouge2"].append(scores["rouge2"].fmeasure)
        rouge_scores["rougeL"].append(scores["rougeL"].fmeasure)

    avg_rouge_scores = {metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()}

    return bleu_score, avg_rouge_scores

bleu_score, rouge_scores = evaluate_metrics(generated_answers, reference_answers)

print(f"BLEU Score: {bleu_score:.4f}")
print("ROUGE Scores:")
for metric, score in rouge_scores.items():
    print(f"  {metric.upper()}: {score:.4f}")

output_file = "/homes/cgaponte/ProgDS/TermProject/baseline_gpt2-large_results.txt"
with open(output_file, "w") as f:
    f.write(f"BLEU Score: {bleu_score:.4f}\n")
    for metric, score in rouge_scores.items():
        f.write(f"{metric.upper()} Score: {score:.4f}\n")

print(f"Results saved to {output_file}")
