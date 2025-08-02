# Bert fine-tuned for 5 epochs on WNLI

# 1. Install/upgrade transformers and datasets (uncomment if needed)
# !pip install --upgrade transformers datasets

# 2. Import packages
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# 3. (Optional) Check versions to make sure
import transformers
import datasets

print("Transformers version:", transformers.__version__)
print("Datasets version:", datasets.__version__)

# Clear cache (optional)
import os
os.system("rm -rf /root/.cache/huggingface/datasets")

# Load WNLI train split
wnli_train = load_dataset("glue", "wnli", split="train")

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize function
def preprocess(batch):
    return tokenizer(batch['sentence1'], batch['sentence2'], truncation=True, padding=True)

# Tokenize dataset
wnli_train_tokenized = wnli_train.map(preprocess, batched=True)

# Set PyTorch format
wnli_train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Training arguments (no evaluation_strategy)
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_steps=10,
    disable_tqdm=False,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=wnli_train_tokenized,
)

# Start training
trainer.train()

# Output (accuracy at various logging steps)
# 10  0.714700
# 20  0.698600
# 30  0.693100
# 40  0.694700
# 50  0.703100
# 60  0.730500
# 70  0.694100
# 80  0.710300
# 90  0.699400
# 100 0.710900
# 110 0.704200
# 120 0.702600
# 130 0.692700
# 140 0.691600
# 150 0.708800
# 160 0.701000
# 170 0.710300
# 180 0.694400
# 190 0.701000
# 200 0.694600
# 210 0.697400
# 220 0.691700
# 230 0.706000
# 240 0.694200
# 250 0.699400
# 260 0.701700
# 270 0.683900
# 280 0.705000
# 290 0.705700
# 300 0.697700
# 310 0.704900
# 320 0.716100
# 330 0.692400
# 340 0.683800
# 350 0.699800
# 360 0.694400
# 370 0.697100
# 380 0.697700
# 390 0.686500
# 400 0.686900
