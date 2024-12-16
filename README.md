# olympics_chatbot
Chatbot trained on olympics dataset!

Original dataset collected from: https://www.kaggle.com/datasets/piterfm/paris-2024-olympic-summer-games

Data preparation steps:
First downloaded my dataset from kaggle. Original data is in olympics_original folder.

Next, processed with my Preprocessing_Scripts.ipynb to normalize data, remove unnecessary or redundant columns, select which datasets to keep. This data is in First_Processed_Olympics_data.zip

Next, converted data to QA format using Question_Formatting.ipynb, where I normalized the data and the answers, generated questions for each dataset, and converted data to appropriate context/question/answer format for training a QA model. This data is in QA_Datasets.zip

Lastly, used Combine_Datasets.ipynb to combine all QA formatted datasets into one large dataset. This finalized dataset is combined_qa_data.zip.



Below is a run-down of all files in this repository (both working and non-working scripts):

Working Scripts:

Data Wrangling/Evaluation/Visualization .ipynb Scripts:

Preprocessing_Scripts.ipynb - First processing of the raw dataset. Removed unnecessary columns and datasets, converted rows to normalized strings

Question_Formatting.ipynb - Converting pre-processed datsets to QA datasets. Generated questions, parsed fields, set data up in standard QA format: context, question, answer

Combine_Datasets.ipynb - Combining all the scripts I individually formatted to QA datsets

plot_losses.ipynb - Plotting training loss curves for most of my trained models

hypothesis_testing_script.ipynb - Back-generating data for ROUGE scores, comparing baseline gpt2-large model to FLAN-T5 model, extracting P-Value




Fine-Tuning Scripts:

albert_fine_tune.py - Fine tuning ALBERT model on my Olympics dataset

fine_tune_BERTS.py - Fine tuning RoBERTa model on my Olympics dataset

just_bert_finetune.py - Fine tuning BERT model on my Olympics dataset



Fine_Tune_gpt2-large.py - Fine tuning gpt2-large on my Olympics dataset

gpt2_large_Qlora.py - Fine tuning gpt2-large on full SQuAD dataset

squad_gpt2-large_finetune.py - Fine tuning gpt2-large that was pre-trained with SQuAD dataset on my Olympics dataset

Fine_Tune_gpt2-xl.py - Fine tuning gpt2-xl on my Olympics dataset



Flan-t5_FineTune.py - Fine tuning FLAN-T5 on my Olympics dataset





Evaluation:

baseline_gpt2-large_evaluation.py - Evaluation for baseline, un-trained gpt2-large model

flan-t5_eval.py - Evaluation for FLAN-T5 model trained on my Olympics data



Non-Working Scripts:

albert_eval.py - Evaluating fine-tuned ALBERT

just_bert_eval.py - Evaluating fine-tuned BERT

roberta_eval.py - Evaluating fine-tuned RoBERTa

eval_gpt2-large.py - Evaluating fine-tuned gpt2-large

eval_gpt2-xl.py - Evaluating fine-tuned gpt2-xl


DATASETS:
olympics_original - All raw, original datasets. Retrieved from Kaggle.
First_Processed_Olympics_data.zip - All datasets after first processing pass
QA_Datasets.zip - All datasets after converting to QA format
combined_qa_data.zip - Final dataset! All QA datsets combined to one. This is what the models trained on.
