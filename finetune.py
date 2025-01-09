from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

# Load your Q&A dataset
dataset = load_dataset("json", data_files="/content/new.json")

# Ensure dataset keys are correctly renamed
def rename_keys(example):
    return {"prompt": example["question"], "response": example["answer"]}
dataset = dataset.map(rename_keys)

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_tinyllama",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    warmup_steps=500,
    learning_rate=3e-5,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=500,
    save_total_limit=3,
    fp16=True,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Add a padding token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token_id = tokenizer.eos_token_id  # Set pad token id to eos token id
model.resize_token_embeddings(len(tokenizer))

def tokenize_function(example):
    # Tokenize the prompt and response
    prompt = tokenizer(example["prompt"], padding="max_length", truncation=True, max_length=128)
    response = tokenizer(example["response"], padding="max_length", truncation=True, max_length=128)

    # Create labels by shifting the response input_ids
    labels = response['input_ids']
    labels = [tokenizer.pad_token_id] + labels[:-1]  # Shift labels to the right

    return {
        "input_ids": torch.tensor(prompt["input_ids"]),
        "attention_mask": torch.tensor(prompt["attention_mask"]),
        "labels": torch.tensor(labels),
        "label_attention_mask": torch.tensor(response["attention_mask"])  # For calculating loss
    }

# Split dataset into train and eval sets
train_test_split = dataset["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=False)
tokenized_eval = eval_dataset.map(tokenize_function, batched=False)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval  # Pass the eval dataset here
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_tinyllama")
tokenizer.save_pretrained("./fine_tuned_tinyllama")
