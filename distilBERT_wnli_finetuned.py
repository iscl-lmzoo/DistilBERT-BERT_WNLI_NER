# DistilBERT fine-tuned for 5 epochs on WNLI dataset

# 1. Install dependencies (uncomment if needed)
# !pip install --upgrade transformers datasets

# 2. Import packages
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import transformers
import datasets

# 3. Print versions
print("Transformers version:", transformers.__version__)
print("Datasets version:", datasets.__version__)

# 4. Optional: clear HF cache (uncomment if needed)
# import os
# os.system("rm -rf /root/.cache/huggingface/datasets")

# 5. Load WNLI dataset
wnli_train = load_dataset("glue", "wnli", split="train")

# 6. Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 7. Preprocessing function
def preprocess(batch):
    return tokenizer(batch['sentence1'], batch['sentence2'], truncation=True, padding=True)

# 8. Tokenize dataset
wnli_train_tokenized = wnli_train.map(preprocess, batched=True)
wnli_train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 9. Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    logging_steps=10,
    save_steps=10,
    disable_tqdm=False,
)

# 10. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=wnli_train_tokenized,
)

# 11. Start training
trainer.train()

# Step Training Loss (logged every 10 steps)
# 10   0.697400
# 20   0.708300
# 30   0.697100
# 40   0.701700
# 50   0.702800
# 60   0.706400
# 70   0.691200
# 80   0.700300
# 90   0.700900
# 100  0.697200
# 110  0.702600
# 120  0.702000
# 130  0.690400
# 140  0.687500
# 150  0.708700
# 160  0.694400
# 170  0.691000
# 180  0.695700
# 190  0.697400
# 200  0.695300
# 210  0.702400
# 220  0.687600
# 230  0.691200
# 240  0.690700
# 250  0.700300
# 260  0.691400
# 270  0.700600
# 280  0.698200
# 290  0.698300
# 300  0.693800
# 310  0.688800
# 320  0.708300
# 330  0.685400
# 340  0.689300
# 350  0.693600
# 360  0.695100
# 370  0.681900

# Note: Final output showed "400/400 2:51:17, Epoch 5/5" — this confirms all training steps completed.
# No warning observed, just standard training completion log.

