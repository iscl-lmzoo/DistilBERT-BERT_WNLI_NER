
!pip install transformers
!pip install --upgrade transformers

from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/My Drive/train_complete.csv'


with open(file_path, 'r') as f:
    for _ in range(5):
        print(f.readline().strip())

from datasets import load_dataset

data_files = {'train': '/content/drive/My Drive/train_complete.csv'}
dataset = load_dataset('csv', data_files=data_files, split='train')

print(dataset.column_names)
print(dataset[0])

# Output:
# premise,hypothesis,label,len_P,len_H,len_input,P_H_vocab_intersection
# "I stuck a pin through a carrot. When I pulled the pin out, it had a hole.",The
# John couldn't see the stage with Billy in front of him because he is so short.
# The police arrested all of the gang members. They were trying to stop the drug
# Steve follows Fred's example in everything. He influences him hugely.,Steve

# next output ['premise', 'hypothesis', 'label', 'len_P', 'len_H', 'len_input', 'P_H_vocab_intersection']
# {'premise': 'I stuck a pin through a carrot. When I pulled the pin out, it}





from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import accuracy_score

data_files = {'test': '/content/drive/My Drive/train_complete.csv'}  # Your file
dataset = load_dataset('csv', data_files=data_files, split='test')

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

label_map = {'entailment': 0, 'not_entailment': 1}

true_labels = [label_map[ex['label']] for ex in dataset]

inputs = tokenizer(
    dataset['premise'],
    dataset['hypothesis'],
    padding=True,
    truncation=True,
    return_tensors='pt'
)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()

accuracy = accuracy_score(true_labels, predictions)
print(f"Zero-shot accuracy on WNLI: {accuracy:.4f}")

# Zero-shot accuracy on WNLI: 0.4992



from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

# BERT
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

inputs = tokenizer(
    dataset['premise'],
    dataset['hypothesis'],
    padding=True,
    truncation=True,
    return_tensors='pt'
)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).tolist()

accuracy = accuracy_score(true_labels, predictions)
print(f"Zero-shot accuracy on WNLI (BERT): {accuracy:.4f}")

# Zero-shot accuracy on WNLI (BERT): 0.5087


# Zero-shot accuracy on WNLI: 0.4992
# Zero-shot accuracy on WNLI (BERT): 0.5087
