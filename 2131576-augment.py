'''
Enrico M. Aldorasi - 2131576
MNLP Homework 2
Adversarial Natural Language Inference 
Augmentation script
'''

import json
import random
from datasets import load_dataset
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
import os
import zipfile

corpora_dir = nltk.data.find("corpora")
wordnet_zip = os.path.join(corpora_dir, "wordnet.zip")

if os.path.exists(wordnet_zip):
    with zipfile.ZipFile(wordnet_zip, 'r') as zip_ref:
        zip_ref.extractall(corpora_dir)
    print("Manually extracted WordNet data")
else:
    print("WordNet zip file not found")

# Check again
if 'wordnet' in os.listdir(corpora_dir):
    print("WordNet is now available")
else:
    print("WordNet is still not available")
nltk.download('averaged_perceptron_tagger')

# Cache WordNet lookups
noun_cache = {}
verb_cache = {}

# Initialize summarization model
print("Initializing BART model for summarization...")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

def get_hypernym(word):
    # Get a hypernym for a given word
    if word not in noun_cache:
        synsets = wordnet.synsets(word, pos=wordnet.NOUN)
        if synsets:
            hypernyms = synsets[0].hypernyms()
            if hypernyms:
                noun_cache[word] = hypernyms[0].lemmas()[0].name()
            else:
                noun_cache[word] = word
        else:
            noun_cache[word] = word
    return noun_cache[word]

def get_synonym(word):
    # Get a synonym for a given word
    if word not in verb_cache:
        synsets = wordnet.synsets(word, pos=wordnet.VERB)
        if synsets:
            synonyms = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
            if synonyms:
                verb_cache[word] = random.choice(synonyms)
            else:
                verb_cache[word] = word
        else:
            verb_cache[word] = word
    return verb_cache[word]

def negate_verb(sentence):
    # Negate the first verb in the sentence
    words = word_tokenize(sentence)
    tagged = pos_tag(words)
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith('VB'):
            if word.lower() in ['is', 'are', 'was', 'were']:
                words[i] += " not"
            elif word.lower() == 'have':
                words[i] = "haven't"
            elif word.lower() == 'has':
                words[i] = "hasn't"
            else:
                words[i] = "didn't " + word
            break
    return ' '.join(words)

def summarize_text(text):
    # Summarize the given text using BART model
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], num_beams=2, min_length=0, max_length=50)
    return tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def augment_sample(sample):
    # Augment a single sample using one of the techniques
    augmented = {k: v for k, v in sample.items()}
    
    technique = random.choice(['summarize', 'hypernym', 'synonym', 'negate'])
    
    if technique == 'summarize':
        augmented['premise'] = summarize_text(sample['premise'])
    elif technique == 'hypernym':
        words = word_tokenize(sample['hypothesis'])
        tagged = pos_tag(words)
        for i, (word, tag) in enumerate(tagged):
            if tag.startswith('NN'):
                words[i] = get_hypernym(word)
                break
        augmented['hypothesis'] = ' '.join(words)
    elif technique == 'synonym':
        words = word_tokenize(sample['hypothesis'])
        tagged = pos_tag(words)
        for i, (word, tag) in enumerate(tagged):
            if tag.startswith('VB'):
                words[i] = get_synonym(word)
                break
        augmented['hypothesis'] = ' '.join(words)
    elif technique == 'negate':
        augmented['hypothesis'] = negate_verb(sample['hypothesis'])
        if augmented['label'] == 'ENTAILMENT':
            augmented['label'] = 'CONTRADICTION'
        elif augmented['label'] == 'CONTRADICTION':
            augmented['label'] = 'ENTAILMENT'
    
    augmented['augmented'] = True
    augmented['technique'] = technique
    return augmented

def process_batch(batch):
    # Process a batch of samples
    results = []
    for i in range(len(batch['id'])):
        sample = {
            'id': batch['id'][i],
            'premise': batch['premise'][i],
            'hypothesis': batch['hypothesis'][i],
            'label': batch['label'][i],
            'wsd': batch['wsd'][i],
            'srl': batch['srl'][i]
        }
        original_sample = {**sample, 'augmented': False, 'technique': 'original'}
        augmented_sample = augment_sample(sample)
        results.extend([original_sample, augmented_sample])
    return results

def main():
    print("Loading dataset...")
    dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli", split="train")
    
    batch_size = 100
    augmentation_techniques = {'summarize': 0, 'hypernym': 0, 'synonym': 0, 'negate': 0, 'original': 0}
    label_distribution = {'ENTAILMENT': 0, 'CONTRADICTION': 0, 'NEUTRAL': 0}
    
    print("Starting augmentation process...")
    with open("augmented.jsonl", "w") as f:
        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing batches"):
            batch = dataset[i:i+batch_size]
            augmented_batch = process_batch(batch)
            for item in augmented_batch:
                json.dump(item, f)
                f.write("\n")
                augmentation_techniques[item['technique']] += 1
                label_distribution[item['label']] += 1

    print("Augmented dataset saved to 'augmented.jsonl'")

    # Print statistics
    print("\nAugmentation Technique Distribution:")
    for technique, count in augmentation_techniques.items():
        print(f"{technique}: {count}")

    print("\nLabel Distribution:")
    for label, count in label_distribution.items():
        print(f"{label}: {count}")

    # Plot augmentation technique distribution
    plt.figure(figsize=(10, 5))
    plt.bar(augmentation_techniques.keys(), augmentation_techniques.values())
    plt.title("Augmentation Technique Distribution")
    plt.xlabel("Technique")
    plt.ylabel("Count")
    plt.savefig("augmentation_techniques.png")
    print("Augmentation technique distribution plot saved as 'augmentation_techniques.png'")

    # Plot label distribution
    plt.figure(figsize=(10, 5))
    plt.bar(label_distribution.keys(), label_distribution.values())
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("label_distribution.png")
    print("Label distribution plot saved as 'label_distribution.png'")

    # Print example results
    print("\nExample Results:")
    with open("augmented.jsonl", "r") as f:
        for _ in range(5):
            example = json.loads(f.readline())
            print(f"ID: {example['id']}")
            print(f"Technique: {example['technique']}")
            print(f"Premise: {example['premise'][:100]}...")
            print(f"Hypothesis: {example['hypothesis']}")
            print(f"Label: {example['label']}")
            print("---")

if __name__ == "__main__":
    main()