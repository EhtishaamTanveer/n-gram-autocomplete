# N-gram Based Autocomplete Language Model - Natural Language Processing

This repository contains an implementation of an N-gram based language model for text autocompletion. The model predicts the next word in a sequence based on the preceding N words. This project covers data loading, preprocessing, model training, and evaluation using perplexity metric.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Implementation](#model-implementation)
- [Evaluation](#evaluation)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Future Work](#future-work)

## Introduction

Language models are a fundamental component of many Natural Language Processing (NLP) applications, including speech recognition, machine translation, and text generation. This project focuses on building an N-gram language model, a statistical approach that predicts the probability of a word given the previous N-1 words. This model is then used to provide autocompletion suggestions.

## Features

- **Data Loading and Preprocessing:** Handles loading text data from files and performs essential preprocessing steps like sentence splitting, tokenization, vocabulary creation, and handling out-of-vocabulary (OOV) words.
- **N-Gram Model Training:** Constructs count and probability matrices for N-grams (unigrams, bigrams, trigrams, etc.).
- **Probability Estimation:** Uses the probability matrix to estimate the likelihood of word sequences.
- **Autocompletion:** Provides word suggestions based on the input text.
- **Perplexity Evaluation:** Measures the model's performance using the perplexity metric.
- **Configurable N-gram Order:** Easily adjustable N value for different model complexities.

## Installation

1.  Clone the repository:

    ```bash
    git clone [https://github.com/](https://github.com/)<EhtishaamTanveer/n-gram-autocomplete.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd n-gram-autocomplete
    ```

3.  Create a virtual environment:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate      # On Windows
    ```

## Usage

1.  Prepare your training data. The data should be a plain text file (.txt). Place it in a `data` folder.

2.  Run the main script:

    ```bash
    python main.py --n <n_gram_order> --data <path_to_data>
    ```
    For example:
    ```bash
    python main.py --n 3 --data data/my_corpus.txt
    ```

3. You can also use the model interactively for autocompletion:
    ```bash
    python interactive_autocomplete.py --model_path <path_to_saved_model>
    ```

## Data Preprocessing

The preprocessing steps include:

-   **Sentence Splitting:** Dividing the input text into individual sentences.
-   **Tokenization:** Breaking down sentences into individual words or tokens.
-   **Vocabulary Creation:** Building a vocabulary of unique words from the corpus.
-   **OOV Handling:** Replacing words not present in the vocabulary with a special `<UNK>` (unknown) token.

## Model Implementation

The N-gram model is implemented by:

1.  Counting the occurrences of N-grams in the preprocessed data.
2.  Calculating the probabilities of N-grams based on their counts.
3.  Implementing add-k smoothing to avoid division by zero and occurence of very small probabilities.
4.  Storing these probabilities in a probability matrix for efficient lookups during autocompletion.

## Evaluation

The model's performance is evaluated using perplexity. Lower perplexity scores indicate better performance. Perplexity measures how well the model predicts a sample.

## Dependencies

-   Python 3.x
-   Numpy
-   NLTK
-   Pandas
-   Math
-   Random

## File Structure

* n-gram-autocomplete/
  * data/
    * en_US.twitter.txt: Tweets Corpus
  * n-gram-autocomplete.ipynb: Main Jupyter Notebook containing project code
  * w3_unittest.py: Python file containing the test cases to check functions in the main notebook
 
## Future Work

-   Explore more advanced language models (e.g., recurrent neural networks).
-   Implement caching mechanisms to speed up autocompletion.

## Acknowledgements

* [Coursera](https://www.coursera.org/)
* [Twitter](https://www.x.com/)
