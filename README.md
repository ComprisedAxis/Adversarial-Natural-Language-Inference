# Adversarial Natural Language Inference 3-class Classification task
This project focuses on Natural Language Inference (NLI) using transformer-based models and data augmentation techniques. It consists of two main components: a data augmentation script and a main training/testing script.

Table of Contents:
1. Requirements
2. Data Augmentation
3. Training and Testing
4. Evaluation

1. Requirements
Python 3.7+
PyTorch
Transformers
NLTK
Datasets
Scikit-learn
Matplotlib
Seaborn
tqdm

Install the required packages using:
> pip install -r requirements.txt

2. Data Augmentation
The 2131576-augment.py script performs data augmentation on the FEVER NLI dataset. It uses various techniques:

Summarization (using BART)
Hypernym replacement
Synonym replacement
Sentence negation

The augmented data is saved in a JSONL file (augmented.jsonl).
Run it using:
> python 2131576-augment.py

3. Training and Testing
The 2131576-main.py script handles both training and testing of the NLI models. It supports:

Multiple model architectures (RoBERTa-base, DeBERTa-v3-large)
Plain and enhanced architectures
Training on original or augmented data

To train one of the models on original data:
> python 2131576-main.py train --data [original] --model [roberta/deberta] --architecture [plain/enhanced]

To train one of the models on the Augmented data:
> python 2131576-main.py train --data [augmented] --augmented data [path/to/augmented.jsonl] --model [roberta/deberta] --architecture [plain/enhanced]

Testing on original or adversarial data:
> python 2131576-main.py test --data [original/adversarial]

4. Evaluation
The 2131576-main.py script evaluates the model using:

Accuracy
Precision
Recall
F1 Score

It also generates:

Training progress plots
Confusion matrix

Results are printed to the console and saved as images.