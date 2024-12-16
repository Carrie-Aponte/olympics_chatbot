# Evaluating my gpt2-xl that i fine-tuned on my Olympics dataset

import bitsandbytes as bnb
import torch
import pandas as pd

from transformers import GPT2Tokenizer, GPT2LMHeadModel, BitsAndBytesConfig, GenerationConfig, TFTrainingArguments, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split
import nltk
from rouge_score import rouge_scorer
from bert_score import score
from sacrebleu.metrics import BLEU
from torch.amp.autocast_mode import autocast

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


model_path = "/homes/cgaponte/ProgDS/TermProject/gpt2-xl_olympics_finetuned_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = GPT2LMHeadModel.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    low_cpu_mem_usage=True  
).to(device)


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
    input_prompt = f"{examples['context']}\n### Question: {examples['question']}\n### Response: {examples['answer']}"
    return {"text": input_prompt}

mapped_dataset = dataset.map(
    input_tokenizer, 
    remove_columns=dataset["train"].column_names, 
    load_from_cache_file=False
) # mapping dataset to be the right format

#mapped_dataset.save_to_disk("/homes/cgaponte/ProgDS/TermProject/mapped_olympics_dataset")


def generate_answers(dataset, model, tokenizer, batch_size=4, max_length=350):
    generated_answers = []
    actual_answers = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        
        
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=50,
                num_beams=3,
                #early_stopping=True,
                do_sample=False,
                temperature=0.7,
                top_p=0.95,
            )


        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_answers.extend(batch_predictions)


        batch_actuals = [text.split("### Response:")[-1].strip() for text in batch["text"]]
        actual_answers.extend(batch_actuals)

    return generated_answers, actual_answers


def save_predictions(generated_answers, actual_answers, output_file="gpt2-xl_generated_vs_actual.txt"):
    with open(output_file, "w") as f:
        for i, (gen, actual) in enumerate(zip(generated_answers, actual_answers)):
            f.write(f"Example {i + 1}:\n")
            f.write(f"Generated: {gen}\n")
            f.write(f"Actual: {actual}\n")
            f.write("-" * 50 + "\n")
    print(f"Generated vs. Actual answers saved to {output_file}")


def evaluate_metrics(generated_answers, actual_answers, output_file="gpt2-xl_evaluation_metrics.txt"):
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
validation_dataset = validation_dataset.select(range(0, 1000))

print("Generating answers...")
generated_answers, actual_answers = generate_answers(validation_dataset, model, tokenizer)

save_predictions(generated_answers, actual_answers)
evaluate_metrics(generated_answers, actual_answers)