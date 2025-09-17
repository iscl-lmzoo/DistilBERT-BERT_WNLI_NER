# pip install transformers --quiet
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'GPU' if device == 0 else 'CPU'}")

bert_model_name = "dslim/bert-base-NER"
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForTokenClassification.from_pretrained(bert_model_name)
bert_ner = pipeline("ner", model=bert_model, tokenizer=bert_tokenizer, device=device)

distilbert_model_name = "elastic/distilbert-base-uncased-finetuned-conll03-english"
distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_name)
distilbert_model = AutoModelForTokenClassification.from_pretrained(distilbert_model_name)
distilbert_ner = pipeline("ner", model=distilbert_model, tokenizer=distilbert_tokenizer, device=device)

# Test sentences
sentences = [
    "Hugging Face Inc. is a company based in New York City.",
    "My name is Sarah and I live in London.",
    "Apple is looking at buying U.K. startup for $1 billion."
]

print("\nModel: BERT NER")
for i, sentence in enumerate(sentences, 1):
    entities = bert_ner(sentence)
    print(f"Sentence {i}: {sentence}")
    print(f"Entities: {entities}\n")

print("\nModel: DistilBERT NER")
for i, sentence in enumerate(sentences, 1):
    entities = distilbert_ner(sentence)
    print(f"Sentence {i}: {sentence}")
    print(f"Entities: {entities}\n")


from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

examples = [
    {"sentence": "Barack Obama was born in Hawaii.", "entities": [("Barack Obama", "PER"), ("Hawaii", "LOC")]},
    {"sentence": "Google is headquartered in Mountain View.", "entities": [("Google", "ORG"), ("Mountain View", "LOC")]},
    {"sentence": "Angela Merkel was Chancellor of Germany.", "entities": [("Angela Merkel", "PER"), ("Germany", "LOC")]},
    {"sentence": "Elon Musk founded SpaceX in California.", "entities": [("Elon Musk", "PER"), ("SpaceX", "ORG"), ("California", "LOC")]},
    {"sentence": "The United Nations is in New York.", "entities": [("United Nations", "ORG"), ("New York", "LOC")]},
    {"sentence": "Amazon has offices in Seattle.", "entities": [("Amazon", "ORG"), ("Seattle", "LOC")]},
    {"sentence": "Cristiano Ronaldo plays for Al-Nassr.", "entities": [("Cristiano Ronaldo", "PER"), ("Al-Nassr", "ORG")]},
    {"sentence": "Meta Platforms owns Instagram.", "entities": [("Meta Platforms", "ORG"), ("Instagram", "ORG")]},
    {"sentence": "Lionel Messi joined Inter Miami.", "entities": [("Lionel Messi", "PER"), ("Inter Miami", "ORG")]},
    {"sentence": "Apple was founded in Cupertino.", "entities": [("Apple", "ORG"), ("Cupertino", "LOC")]},
    {"sentence": "Boris Johnson is from the UK.", "entities": [("Boris Johnson", "PER"), ("UK", "LOC")]},
    {"sentence": "NASA launched Artemis from Florida.", "entities": [("NASA", "ORG"), ("Artemis", "MISC"), ("Florida", "LOC")]},
    {"sentence": "The Louvre is in Paris.", "entities": [("Louvre", "ORG"), ("Paris", "LOC")]},
    {"sentence": "Samsung is based in Seoul.", "entities": [("Samsung", "ORG"), ("Seoul", "LOC")]},
    {"sentence": "The World Health Organization warned about COVID-19.", "entities": [("World Health Organization", "ORG"), ("COVID-19", "MISC")]},
    {"sentence": "Taylor Swift performed in London.", "entities": [("Taylor Swift", "PER"), ("London", "LOC")]},
    {"sentence": "Netflix streamed a show from Los Angeles.", "entities": [("Netflix", "ORG"), ("Los Angeles", "LOC")]},
    {"sentence": "Pfizer developed the vaccine with BioNTech.", "entities": [("Pfizer", "ORG"), ("BioNTech", "ORG")]},
    {"sentence": "Jack Ma started Alibaba in China.", "entities": [("Jack Ma", "PER"), ("Alibaba", "ORG"), ("China", "LOC")]},
    {"sentence": "The BBC is a British broadcaster.", "entities": [("BBC", "ORG")]},
    {"sentence": "Harvard University is in Cambridge.", "entities": [("Harvard University", "ORG"), ("Cambridge", "LOC")]},
    {"sentence": "The G7 summit was held in Japan.", "entities": [("G7", "ORG"), ("Japan", "LOC")]},
    {"sentence": "The President lives in the White House.", "entities": [("White House", "LOC")]},
    {"sentence": "Spotify is available in India.", "entities": [("Spotify", "ORG"), ("India", "LOC")]},
    {"sentence": "Satya Nadella is the CEO of Microsoft.", "entities": [("Satya Nadella", "PER"), ("Microsoft", "ORG")]},
    {"sentence": "IBM researchers work in Zurich.", "entities": [("IBM", "ORG"), ("Zurich", "LOC")]},
    {"sentence": "Jeff Bezos founded Blue Origin.", "entities": [("Jeff Bezos", "PER"), ("Blue Origin", "ORG")]},
    {"sentence": "Google Maps is used in Germany.", "entities": [("Google Maps", "ORG"), ("Germany", "LOC")]},
    {"sentence": "Paris Saint-Germain won the game.", "entities": [("Paris Saint-Germain", "ORG")]},
    {"sentence": "Cambridge Analytica shut down after scandal.", "entities": [("Cambridge Analytica", "ORG")]},
]

models = {
    "DistilBERT": pipeline("ner", model="elastic/distilbert-base-uncased-finetuned-conll03-english", aggregation_strategy="simple", device=device),
    "BERT": pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=device),
}

def evaluate_model(pipeline, examples):
    true_labels = []
    pred_labels = []
    for ex in examples:
        text = ex["sentence"]
        true_ents = set((ent[0].lower(), ent[1]) for ent in ex["entities"])
        pred_ents_raw = pipeline(text)
        pred_ents = set((ent['word'].lower(), ent['entity_group']) for ent in pred_ents_raw)
        all_ents = true_ents.union(pred_ents)
        for ent in all_ents:
            true_labels.append(1 if ent in true_ents else 0)
            pred_labels.append(1 if ent in pred_ents else 0)
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    accuracy = np.mean([t == p for t, p in zip(true_labels, pred_labels)])
    return accuracy, precision, recall, f1

for name, ner_model in models.items():
    accuracy, precision, recall, f1 = evaluate_model(ner_model, examples)
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
# Device set to use CPU
# Some weights of the model checkpoint at dslim/bert-base-NER were not used when initializing BertForTokenClassification.
# This IS expected if you are initializing BertForTokenClassification from a checkpoint that was not specifically trained for token classification.

# Device set to use CPU

# DistilBERT Results:
# Accuracy: 46.81%
# Precision: 55.00%
# Recall: 75.86%
# F1 Score: 63.77%

# BERT Results:
# Accuracy: 65.33%
# Precision: 74.24%
# Recall: 84.48%
# F1 Score: 79.03%

# Model: BERT NER
# Sentence 1: Hugging Face Inc. is a company based in New York City.
# Entities: [{'entity': 'B-ORG', 'score': ..., 'word': 'Hugging', ...}, ...]

# Sentence 2: My name is Sarah and I live in London.
# Entities: [{'entity': 'B-PER', 'word': 'Sarah'}, {'entity': 'B-LOC', 'word': 'London'}, ...]

# Sentence 3: Apple is looking at buying U.K. startup for $1 billion.
# Entities: [{'entity': 'B-ORG', 'word': 'Apple'}, {'entity': 'B-LOC', 'word': 'U.K.'}, ...]

# Model: DistilBERT NER
# Sentence 1: Hugging Face Inc. is a company based in New York City.
# Entities: [{'entity': 'B-ORG', 'score': ..., 'word': 'Hugging', ...}, ...]

# Sentence 2: My name is Sarah and I live in London.
# Entities: [{'entity': 'B-PER', 'word': 'Sarah'}, {'entity': 'B-LOC', 'word': 'London'}, ...]

# Sentence 3: Apple is looking at buying U.K. startup for $1 billion.
# Entities: [{'entity': 'B-ORG', 'word': 'Apple'}, {'entity': 'B-LOC', 'word': 'U.K.'}, ...]

