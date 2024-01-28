# train a bert tokenizer
from tokenizers import BertWordPieceTokenizer
from datasets import load_dataset
from transformers import BertTokenizerFast
from transformers import BertConfig, BertForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

# tokenizer
print("==== train bert tokenizer ====")
tokenizer = BertWordPieceTokenizer()

tokenizer.train(
    files=["./dataset/daily_dialogue.txt"],
    vocab_size=30000,
    min_frequency=3,
    show_progress=True,
)

tokenizer.save_model("./bert-babylm-demo")

print(f"trained tokenizer vocab size: {tokenizer.get_vocab_size()}")

# dataset
print("==== prepare dataset ====")
dataset = load_dataset("text", data_files={"train": ["./dataset/daily_dialogue.txt", "./dataset/bnc_spoken.txt"]}, split="train").train_test_split(test_size=0.1, shuffle=True, seed=42)
train_dataset, test_dataset = dataset["train"], dataset["test"]

tokenizer = BertTokenizerFast.from_pretrained("./bert-babylm-demo", max_len=512)

train_dataset = dataset['train'].map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
test_dataset = dataset['test'].map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)

# model config
config = BertConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = BertForMaskedLM(config=config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./Bert",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    learning_rate=1e-4,
    weight_decay=0.01,
    report_to="none",
    fp16=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# training and save model
print("==== training ====")
trainer.train()
trainer.save_model("./Bert")
