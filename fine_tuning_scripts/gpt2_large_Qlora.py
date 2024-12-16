# fine tuning gpt2-large on full SQuAD dataset!

import bitsandbytes as bnb
from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from trl import SFTTrainer

import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_name = "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

tokenizer.pad_token = tokenizer.eos_token 


dataset_name = "squad"

dataset = load_dataset(dataset_name)

def input_tokenizer(examples):
    input_prompt = (
        examples['context']
        + "\n### Question: " + examples['question']
        + "\n### Response: " + examples['answers']['text'][0]
    )
    return {'text': input_prompt}

mapped_dataset = dataset.map(input_tokenizer, load_from_cache_file=False)

lora_config = LoraConfig(
    r=16,  
    lora_alpha=32,
    lora_dropout=0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    remove_unused_columns=True,
    evaluation_strategy="epoch",
    report_to="none",
    fp16=True 
)


trainer = SFTTrainer(
    model,
    train_dataset=mapped_dataset['train'],  
    eval_dataset=mapped_dataset['validation'],  
    args=training_args,
    tokenizer=tokenizer,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=512
)


sample = dataset['validation'][0]

input_text = sample['context'] + '\n' + sample['question']

input_encodings = tokenizer(
    input_text,
    truncation=True,
    max_length=512,
    padding='max_length',
    return_tensors='pt'
)

input_ids = input_encodings['input_ids'].to(model.device)
attention_mask = input_encodings['attention_mask'].to(model.device)

outputs = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=128,  
    temperature=0.3,  
    top_p=0.9,  
    num_beams=5,  
    early_stopping=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Generated Answer: {generated_text}")
print(f"Original Answer: {sample['answers']['text']}")
