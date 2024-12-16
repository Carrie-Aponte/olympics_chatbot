# Evaluating my gpt2-large that i fine-tuned on my Olympics dataset

import bitsandbytes as bnb
import torch
import pandas as pd

from transformers import GPT2Tokenizer, GPT2LMHeadModel, BitsAndBytesConfig, GenerationConfig, TFTrainingArguments, TextDataset
from transformers import Trainer, TrainingArguments, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
import nltk
from rouge_score import rouge_scorer
from bert_score import score
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader

nltk.download("punkt")
nltk.download("wordnet")


model_path = "/homes/cgaponte/ProgDS/TermProject/gpt2-large_olympics_finetuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path).to("cuda")
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

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

mapped_dataset = dataset.map(
    input_tokenizer, 
    remove_columns=dataset["train"].column_names, 
    load_from_cache_file=False
) # mapping dataset to be the right format

mapped_dataset.save_to_disk("/homes/cgaponte/ProgDS/TermProject/mapped_olympics_dataset")


def generate_answers(dataset, model, tokenizer, max_length=512, batch_size=16):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    generated_answers = []
    actual_answers = []

    for batch in dataloader:
        input_prompts = batch["text"]
        inputs = tokenizer(
            input_prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        ).to(model.device)

        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            early_stopping=True,
            do_sample = True,
        )
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        generated_answers.extend(generated_texts)
        actual_answers.extend(
            [input_prompt.split("### Response:")[-1].strip() for input_prompt in input_prompts]
        )

    return generated_answers, actual_answers


def save_predictions(generated_answers, actual_answers, output_file="gpt2-large_generated_vs_actual.txt"):
    with open(output_file, "w") as f:
        for i, (gen, actual) in enumerate(zip(generated_answers, actual_answers)):
            f.write(f"Example {i + 1}:\n")
            f.write(f"Generated: {gen}\n")
            f.write(f"Actual: {actual}\n")
            f.write("-" * 50 + "\n")
    print(f"Generated vs. Actual answers saved to {output_file}")


def evaluate_metrics(generated_answers, actual_answers, output_file="gpt2-large_evaluation_metrics.txt"):
    bleu = BLEU()
    bleu_score = bleu.corpus_score(generated_answers, [actual_answers]).score

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for gen, actual in zip(generated_answers, actual_answers):
        scores = rouge.score(actual, gen)
        for key in rouge_scores:
            rouge_scores[key].append(scores[key].fmeasure)

    avg_rouge = {key: sum(values) / len(values) for key, values in rouge_scores.items()}

    _, _, bert_scores = score(generated_answers, actual_answers, lang="en", verbose=True)
    avg_bert_score = bert_scores.mean().item()

    with open(output_file, "w") as f:
        f.write(f"BLEU Score: {bleu_score:.4f}\n")
        for key, value in avg_rouge.items():
            f.write(f"Average {key.upper()} Score: {value:.4f}\n")
        f.write(f"Average BERTScore: {avg_bert_score:.4f}\n")

    print(f"Metrics saved to {output_file}")


validation_dataset = mapped_dataset["validation"]

print("Generating answers...")
generated_answers, actual_answers = generate_answers(validation_dataset, model, tokenizer)

save_predictions(generated_answers, actual_answers)
evaluate_metrics(generated_answers, actual_answers)