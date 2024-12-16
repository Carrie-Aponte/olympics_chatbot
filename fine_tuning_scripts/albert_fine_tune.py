################### THE SOURCE CODE FOR THE FINE-TUNING OF THE BERT MODELS CAME FROM ANOTHER STUDENT. I HEAVILY MODIFIED THE PROGRAM TO FIT MY NEEDS AND TRAIN ON MY SPECIFIC DATA. ##########################

# training bert models on my olympics dataset :)

import argparse
import os
import warnings
import torch
import pandas as pd
import collections
import nltk
import matplotlib.pyplot as plt
from statistics import mean
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sacrebleu.metrics import BLEU
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AlbertForQuestionAnswering,
    AlbertTokenizerFast,
    BertForQuestionAnswering,
    BertTokenizerFast,
    RobertaForQuestionAnswering,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    logging as transformers_logging,
)
from transformers.trainer_callback import TrainerCallback
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model

nltk.download("punkt")

warnings.simplefilter(action='ignore', category=FutureWarning)
transformers_logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


config = {
    "train_bert": False,       # train BERT
    "train_roberta": False,   # train RoBERTa
    "train_albert": True,    # train ALBERT
    "epochs": 3,
    "batch_size": 2,
    "learning_rate": 3e-5,
    "max_length": 384,
    "doc_stride": 128,
    "gradient_accumulation_steps": 2,
}

# making sure I only chose one model :)
if sum([config["train_bert"], config["train_roberta"], config["train_albert"]]) != 1:
    raise ValueError("Exactly one model must be selected for training.")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


file_path = "/homes/cgaponte/ProgDS/TermProject/combined_qa_data.csv"
olympics_df = pd.read_csv(file_path)

olympics_df["context"] = olympics_df["context"].str.replace(r"\[.*?\]", "", regex=True)  
olympics_df["context"] = olympics_df["context"].str.replace(r"\|", " ", regex=True)   
olympics_df["context"] = olympics_df["context"].str.replace(r":", " -", regex=True)    


print(olympics_df.isnull().sum())


invalid_rows = olympics_df[
    (olympics_df['context'].str.strip() == '') |
    (olympics_df['question'].str.strip() == '') |
    (olympics_df['answer'].str.strip() == '')
]
print(f"Number of invalid rows: {len(invalid_rows)}")

olympics_df = olympics_df[
    (olympics_df["context"].str.len() > 0) &
    (olympics_df["question"].str.len() > 0) &
    (olympics_df["answer"].str.len() > 0)
]


olympics_df.dropna(subset=['context', 'question', 'answer'], inplace=True)

print(f"Number of rows before filtering, but after dropping n/a rows: {len(olympics_df)}")

olympics_df = olympics_df[
    olympics_df['context'].notnull() &
    olympics_df['question'].notnull() &
    olympics_df['answer'].notnull() &
    olympics_df.apply(lambda row: row['answer'] in row['context'], axis=1)
]

olympics_df = olympics_df[
    olympics_df.apply(lambda row: row['answer'] in row['context'], axis=1)
]

print(f"Remaining rows after filtering: {len(olympics_df)}")


train_df, val_df = train_test_split(olympics_df, test_size=0.2, random_state=42)

train_df['id'] = range(len(train_df))
val_df['id'] = range(len(val_df))


def prepare_hf_dataset(df):
    return Dataset.from_pandas(df)

train_dataset = prepare_hf_dataset(train_df)
val_dataset = prepare_hf_dataset(val_df)

dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

filtered_dataset = dataset.filter(
    lambda example: bool(example["context"]) and 
                    bool(example["question"]) and 
                    bool(example["answer"])
)

max_context_len = 512
max_question_len = 128
max_answer_len = 512
filtered_dataset = filtered_dataset.filter(
    lambda example: len(example["context"]) <= max_context_len and
                    len(example["question"]) <= max_question_len and
                    len(example["answer"]) <= max_answer_len
)


def input_tokenizer(examples):
    context = examples["context"]
    question = examples["question"]
    answer = examples["answer"]
    
    if not context or not question or not answer:
        raise ValueError("Empty context, question, or answer detected.")
    if not answer in context:
        raise ValueError(f"Answer not found in context: {answer}")
    
    input_prompt = f"{context}\n### Question: {question}\n### Response: {answer}"
    return {"text": input_prompt}


mapped_dataset = filtered_dataset.map(
    input_tokenizer, 
    load_from_cache_file=False
)




print(mapped_dataset["train"][0])
print(mapped_dataset["validation"][0])


save_path = "/homes/cgaponte/ProgDS/TermProject/mapped_bert_olympics_dataset"
mapped_dataset.save_to_disk(save_path)

print(f"Mapped dataset saved to {save_path}")


print("Maximum context length:", max(len(c) for c in mapped_dataset["train"]["context"]))
print("Maximum question length:", max(len(q) for q in mapped_dataset["train"]["question"]))
print("Maximum answer length:", max(len(a) for a in mapped_dataset["train"]["answer"]))



def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        max_length=config["max_length"],
        truncation=True,
        padding="max_length",
    )

def preprocess_data(example):
    try:
        tokenized = tokenizer(
            example["context"],
            example["question"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_offsets_mapping=True
        )
        if not tokenized:
            return None
        return tokenized
    except ValueError as e:
        print(f"Skipping invalid row due to error for this example {example}. Error: {e}")
        return None
    
def preprocess_function(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=config["max_length"],
        stride=config["doc_stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    for key in ["input_ids", "attention_mask"]:
        for seq in tokenized_examples[key]:
            if len(seq) != config["max_length"]:
                print(f"Inconsistent {key} length: {len(seq)}")


    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")


    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)


        for i, example in enumerate(tokenized_examples["input_ids"]):
            if len(example) != config["max_length"]:
                print(f"Error: Inconsistent length at index {i}, length: {len(example)}")



        sample_index = sample_mapping[i]
        context = examples["context"][sample_index]
        answer = examples["answer"][sample_index]


        if len(answer) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = context.find(answer)
            end_char = start_char + len(answer)

            if start_char == -1:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                continue


            token_start_index = 0
            while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index -= 1

            token_end_index = len(offsets) - 1
            while token_end_index >= 0 and offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            token_end_index += 1

            if token_start_index < 0 or token_start_index >= len(offsets) or token_end_index < 0 or token_end_index >= len(offsets):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                continue


            if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                tokenized_examples["start_positions"].append(token_start_index)
                tokenized_examples["end_positions"].append(token_end_index)

    return tokenized_examples


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="QUESTION_ANSWERING",
    target_modules=["query", "key", "value"]
)

class DataCollatorForQA(DataCollatorWithPadding):
    def __call__(self, features):
        features = [
            {k: v for k, v in f.items() if k not in ('offset_mapping', 'example_id')}
            for f in features
        ]
        return super().__call__(features)



def train_model(model_name, tokenizer_class, model_class):
    global tokenizer
    
    print(f"\n***** Training {model_name} *****")
    if model_name == 'bert':
        hf_model_name = 'bert-base-uncased'
    elif model_name == 'roberta':
        hf_model_name = 'roberta-base'
    elif model_name == 'albert':
        hf_model_name = 'albert-base-v2'
    else:
        raise ValueError("Unsupported model_name. Use bert, roberta, or albert.")

    tokenizer = tokenizer_class.from_pretrained(hf_model_name)

    
    model = model_class.from_pretrained(hf_model_name)
    model = get_peft_model(model, lora_config)

    tokenized_train_dataset = mapped_dataset["train"].map(
        preprocess_function, batched=True
    )
    tokenized_val_dataset = mapped_dataset["validation"].map(
        prepare_validation_features, batched=True, load_from_cache_file=False
    )


    print("Validating tokenized train dataset lengths...")
    for i, example in enumerate(tokenized_train_dataset):
        tokens = tokenizer(
            example["question"],
            example["context"],
            truncation="only_second",
            max_length=config["max_length"],
            stride=config["doc_stride"],
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        for seq in tokens["input_ids"]:
            if len(seq) != config["max_length"]:
                print(f"Error at index {i}: Length {len(seq)} does not match max_length.")



    tokenized_train_dataset = tokenized_train_dataset.remove_columns(["context", "question", "answer", "text", "__index_level_0__"])

    for row in tokenized_train_dataset:
        if len(row["input_ids"]) != config["max_length"]:
            print(f"Inconsistent input_ids length: {len(row['input_ids'])}")

    tokenized_val_dataset = tokenized_val_dataset.remove_columns(["context", "question", "answer", "text", "__index_level_0__"])

    tokenized_val_dataset_for_eval = tokenized_val_dataset


    print("Tokenized datasets examples:")
    print(tokenized_train_dataset[0])
    print(tokenized_val_dataset[0])

    for i, example in enumerate(tokenized_train_dataset):
        if not isinstance(example['input_ids'], list) or not isinstance(example['attention_mask'], list):
            print(f"Issue in example {i}: {example}")


    for idx, example in enumerate(tokenized_train_dataset):
        if len(example["input_ids"]) != len(example["attention_mask"]):
            print(f"Mismatch at index {idx}: {example}")

    training_args = TrainingArguments(
        output_dir=f"{model_name}_olympics_finetuned",
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        dataloader_num_workers=1,
        disable_tqdm=False,
        remove_unused_columns=True,
        save_strategy='epoch',
    )

    data_collator = DataCollatorForQA(tokenizer, pad_to_multiple_of=8)

    print(f"Number of train examples: {len(tokenized_train_dataset)}")
    print(f"Number of validation examples: {len(tokenized_val_dataset)}")
    print(f"Example tokenized entry: {tokenized_train_dataset[0]}")


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    trainer.save_model(f"{model_name}_olympics_finetuned")
    tokenizer.save_pretrained(f"{model_name}_olympics_finetuned")
    print(f"Model and tokenizer saved to {model_name}_olympics_finetuned.")

    metrics = trainer.evaluate()
    print("Validation metrics:", metrics)

    plot_loss(trainer.state.log_history, model_name)

    evaluate_model(model_name, tokenizer_class, model_class, tokenized_val_dataset_for_eval)




def plot_loss(log_history, model_name):
    train_loss = []
    eval_loss = []
    epochs = []

    for entry in log_history:
        if "loss" in entry: 
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
        if "epoch" in entry: 
            epochs.append(entry["epoch"])

    plt.figure(figsize=(10, 6))
    plt.plot(epochs[:len(train_loss)], train_loss, label="Train Loss", marker="o")
    if eval_loss:
        plt.plot(epochs[:len(eval_loss)], eval_loss, label="Validation Loss", marker="o")

    plt.title(f"Training and Validation Loss for {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_loss_plot.png")
    plt.show()
    print(f"Loss plot saved to {model_name}_loss_plot.png")



def prepare_validation_features(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=config["max_length"],
        stride=config["doc_stride"],
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )


    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["offset_mapping"] = []

    tokenized_examples["example_id"] = []

    for i, offsets in enumerate(offset_mapping):
        sequence_ids = tokenized_examples.sequence_ids(i)
        if not offset_mapping[i]:
            print(f"Warning: Empty offset_mapping for example {i}")
        if sequence_ids is None:
            print(f"Warning: sequence_ids is None for example {i}")

        if sequence_ids is None or offsets is None:
            tokenized_examples["offset_mapping"].append([]) 
            tokenized_examples["example_id"].append(examples["id"][sample_mapping[i]])
            continue

        tokenized_examples["offset_mapping"].append([
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(offsets)
        ])
        tokenized_examples["example_id"].append(examples["id"][sample_mapping[i]])

    return tokenized_examples
    


def evaluate_model(model_name, tokenizer_class, model_class, val_dataset):
    model_path = f"{model_name}_olympics_finetuned"
    tokenizer = tokenizer_class.from_pretrained(model_path)
    model = model_class.from_pretrained(model_path).to(device)


    dataloader = DataLoader(val_dataset, batch_size=4)
    all_start_logits = []
    all_end_logits = []
    features = []

    print("Generating predictions...")
    for batch in dataloader:
        inputs = {key: batch[key].to(device) for key in batch if key in ["input_ids", "attention_mask"]}
        outputs = model(**inputs)

        all_start_logits.append(outputs.start_logits.detach().cpu().numpy())
        all_end_logits.append(outputs.end_logits.detach().cpu().numpy())
        features.append(batch)

        predictions = postprocess_qa_predictions(
            val_dataset, features, (all_start_logits, all_end_logits), tokenizer
        )

        pred_texts = list(predictions.values())
        true_texts = [example["answer"] for example in val_dataset]


    print("Evaluating metrics...")


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

    print("Evaluation complete.")
    print(f"BLEU Score: {bleu_score:.4f}")
    for key, value in avg_rouge.items():
        print(f"{key.upper()} Score: {value:.4f}")
    print(f"BERTScore: {avg_bert_score:.4f}")


    eval_output_file = f"{model_name}_evaluation_metrics.txt"
    with open(eval_output_file, "w") as f:
        f.write(f"BLEU Score: {bleu_score:.4f}\n")
        for key, value in avg_rouge.items():
            f.write(f"Average {key.upper()} Score: {value:.4f}\n")
        f.write(f"Average BERTScore: {avg_bert_score:.4f}\n")
    print(f"Metrics saved to {eval_output_file}")

    gen_output_file = f"{model_name}_generated_vs_actual.txt"
    with open(gen_output_file, "w") as f:
        for i, (gen, actual) in enumerate(zip(pred_texts, true_texts)):
            f.write(f"Example {i + 1}:\n")
            f.write(f"Generated: {gen}\n")
            f.write(f"Actual: {actual}\n")
            f.write("-" * 50 + "\n")
    print(f"Generated vs. Actual answers saved to {gen_output_file}")
    



def postprocess_qa_predictions(examples, features, predictions, tokenizer):
    all_start_logits, all_end_logits = predictions
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

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

            
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
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
                        "text": context[start_char:end_char]
                    })

        
        if valid_answers:
            best_answer = max(valid_answers, key=lambda x: x["score"])
        else:
            best_answer = {"text": "", "score": min_null_score}

        predictions[example["id"]] = best_answer["text"]
    return predictions



# choosing the right model based on what I chose in config :)
if config["train_bert"]:
    train_model('bert', BertTokenizerFast, BertForQuestionAnswering)
elif config["train_roberta"]:
    train_model('roberta', RobertaTokenizerFast, RobertaForQuestionAnswering)
elif config["train_albert"]:
    train_model('albert', AlbertTokenizerFast, AlbertForQuestionAnswering)

