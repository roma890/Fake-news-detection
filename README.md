# Fake-news-detection
This project implements a Fake News Detection system using a multi-head attention-based LSTM model. The model classifies news articles as either "Fake" or "True" based on their textual content. The dataset used consists of labeled news articles.

Dataset
The dataset used for this project consists of labelled news articles. Each article is classified as either Fake or True. The dataset has been split into training (80%) and test(20%) sets.

Dataset source: Fake and Real News Dataset on Kaggle.

Dataset Structure:
True.csv: Contains real news articles.
Fake.csv: Contains fake news articles.
Both files contain the following columns:

title: The headline of the news article.
text: The body of the news article.
label: The label, either "Fake" or "True".

Preprocessing:
The text data undergoes several preprocessing steps to prepare it for the model:
Lowercasing: All text is converted to lowercase.
Stopword Removal: Common stopwords (e.g., "the", "is", "and") are removed using NLTK.
Lemmatization: Words are reduced to their base form (e.g., "running" becomes "run").
Tokenization: Text is tokenized into individual words.
Padding: Each sequence is padded to a fixed length to ensure uniform input to the LSTM model.
Model Architecture
The model architecture is designed to leverage both LSTM and multi-head attention mechanisms to classify the news articles.

Embedding Layer: Maps words to dense vector representations using pre-trained word embeddings (e.g., GloVe or FastText).
Bidirectional LSTM: Captures the sequential nature of text data and learns from both past and future words in the sequence.
Multi-Head Attention Layer: Allows the model to focus on different sections of the text, improving the model's ability to capture important features.
Dense Layers: Fully connected layers for high-level feature learning.

Model Hyperparameters:
Batch size: 32
Epochs: 5
Optimizer: Adam
Learning rate: 0.001
Loss function: Binary cross-entropy

Results After training for 5 epochs, the model achieved the following metrics on the test set: 
Accuracy: 99.47% 
Precision: 99.23% 
Recall: 99.15% 
F1 Score: 99.19% 
The high accuracy and precision indicate strong performance in detecting both real and fake news articles. However, the slight imbalance between false positives and false negatives may require further tuning such as experimentation with learning rates, batch sizes, and architectures.
