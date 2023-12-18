import nltk
nltk.download('brown')
nltk.download('punk')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import brown
from nltk.chunk import RegexpParser

# Tokenize the sentence
sentence = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(sentence)
print("Tokenized Sentence:", tokens)

# Perform Part-of-Speech Tagging
pos_tags = nltk.pos_tag(tokens)
print("Part-of-Speech Tagging:")
print(pos_tags)

# Fetch words from the 'news' category in the Brown corpus
text = brown.words(categories='news')[:1000]

# Generate and analyze bigrams
bigrams = list(ngrams(text, 2))
freq_dist = nltk.FreqDist(bigrams)
print("\nN-gram Analysis (Bigrams with Smoothing):")
for bigram in bigrams:
    print(f"{bigram}: {freq_dist[bigram]}")

# Tag and chunk a new sentence using a defined grammar
tagged_sentence = nltk.pos_tag(word_tokenize("The quick brown fox jumps over the lazy dog"))
grammar = r"NP: {<DT>?<JJ>*<NN>}"
cp = RegexpParser(grammar)
result = cp.parse(tagged_sentence)
print("\nChunking with Regular Expressions and POS tags:")
print(result)
