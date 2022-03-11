# News-Titles-Classification-by-RNN-and-Transformer
You are given a csv file (train.csv) that contains the corresponding label for BBC news with title and content.
In this project, we implement recurrent neural network (RNN, LSTM or GRU) to correctly classify the news in testing data.

## Requirements
Setup packages
```sh
pip install -r requirements.txt
```

## Data structure
The root folder should be structured as follows:
```
  root/
  ├─ data/      # you should download the dataset on the website and set the same name here.
  |  ├─ train.csv
  |  ├─ test.csv
  |  └─ sample_submission.csv
  |
  ├─ README.md
  ├─ RNN.py
  └─ model.py
```

## Text Preprocessing in this projects
1. Choose one kind of tokenizer. how to choose it is depends on which languages. [This]((https://blog.ekbana.com/private-nltk-vs-spacy-3926b3674ee4)) and [this](https://www.analyticsvidhya.com/blog/2019/07/how-get-started-nlp-6-unique-ways-perform-tokenization/) document is  provided for your reference. In this repo, we use nltk.
2. After tokenizing all sentences into lots words, we force all of them to be a lowercase. For examples, "Today" to "today".
3. Lemma, convert all words to their root word. Such as "ran" to "run, "factories" to "factory", or "its" to "it".
4. Remove stop words (as, the ,to, a, all...etc.)
5. Depends on stage of train or test process. While training, we need to get the frequency of each word, and build a vocabularoty using pretrained embedding method (which is illustrated below.), and filter out some uncommon words (like frequency<2). While testing the model, we don't need to filter out any word.


## RNN
We choose a collection of 50 Billion words and represent each word as a 300-dimensional vector glove as initial embedding.
The pretrained embedding code make words which having the similar meaning closer to each other, and make loger distance between the words having opposite meaning (such as wake and sleep).

```sh
from torchtext.vocab import Vocab
vocab = Vocab(counter, min_freq=2, vectors='glove.6B.300d')
```

