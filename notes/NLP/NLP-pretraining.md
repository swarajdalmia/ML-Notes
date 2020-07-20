# NLP - Pretraining 

To understand text, we can begin with its representation, such as treating each word or subword as an individual text token. As we will see in this chapter, the representation of each token can be pretrained on a large corpus, using word2vec, GloVe, or subword embedding models. After pretraining, representation of each token can be a vector, however, it remains the same no matter what the context is. 

- How do embeddings deal with different meaning of words ?

## Word Embeddings (word2vec)
As its name implies, a word vector is a vector used to represent a word. It can also be thought of as the feature vector of a word. The technique of mapping words to vectors of real numbers is also known as word embedding. Over the last few years, word embedding has gradually become basic knowledge in natural language processing.

A couple of great articles
- [https://israelg99.github.io/2017-03-23-Word2Vec-Explained/]
- [https://nathanrooy.github.io/posts/2018-03-22/word2vec-from-scratch-with-python-and-numpy/]

#### Why not use One-hot embeddings 
Although one-hot word vectors are easy to construct, they are usually not a good choice. One of the major reasons is that the one-hot word vectors cannot accurately express the similarity between different words, such as the cosine similarity that we commonly use. 

Word2vec is a tool that we came up with to solve the problem above. It represents each word with a fixed-length vector and uses these vectors to better indicate the similarity and analogy relationships between different words. The Word2vec tool contains two models: skip-gram [Mikolov et al., 2013b] and continuous bag of words (CBOW) [Mikolov et al., 2013a]. Next, we will take a look at the two models and their training methods.

Continuous Bag of Words (CBOW) model can be thought of as learning word embeddings by training a model to predict a word given its context.

Skip-Gram Model is the opposite, learning word embeddings by training a model to predict context given a word.

They also proposed two methods to train the models based on a hierarchical softmax approach or a negative-sampling approach.

Word2Vec is a simple neural network with a single hidden layer, and like all neural networks, it has weights, and during training, its goal is to adjust those weights to reduce a loss function. However, Word2Vec is not going to be used for the task it was trained on, instead, we will just take its hidden weights, use them as our word embeddings, and toss the rest of the model. In the model there are two sets of weight vectors that are trained. The ones from the input to the hidden layer and the one from the hidden layer to the output. After training, they are multiplied, to vocab\*vocab matrix. But is the first matrix, the one that calculates the word embeddings ?

Each of the embeddings for the Skip-gram modela nd the CBOW model are trainined using a neural network. During the training phase, the input to the networks are one-hot word encodings. 
### Skip-Gram Model 
The skip-gram model assumes that a word can be used to generate the words that surround it in a text sequence. For example, we assume that the text sequence is “the”, “man”, “loves”, “his”, and “son”. We use “loves” as the central target word and set the context window size to 2. given the central target word “loves”, the skip-gram model is concerned with the conditional probability for generating the context words, “the”, “man”, “his” and “son”, that are within a distance of no more than 2 words, which is :
```
𝑃("the","man","his","son"∣"loves").
```

We assume that, given the central target word, the context words are generated independently of each other. In this case, the formula above can be rewritten as:
```
𝑃("the"∣"loves")⋅𝑃("man"∣"loves")⋅𝑃("his"∣"loves")⋅𝑃("son"∣"loves")
```

**In the skip-gram model, each word is represented as two dimension vectors, which are used to compute the conditional probability.** Generating the word2vec embedding is done by taking a large corpus of text as it’s input and produces a vector space (feature space), typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space. 

What we finally arrive at is, a matrix of vocab\*vocab which talks about the probability of a word in the context given a specific word in the centre. 

### Continuous Bag of Words

### Comparing the two 
- Statistically, it has the effect that CBOW smoothes over a lot of the distributional information (by treating an entire context as one observation). For the most part, this turns out to be a useful thing for smaller datasets.
- Skip-gram predicts surrounding context words from the target words (inverse of CBOW).
Statistically, skip-gram treats each context-target pair as a new observation, and this tends to do better when we have larger datasetstakestakes.

### Training word2vec
For larger dictionaries with hundreds of thousands or even millions of words, the overhead for computing each gradient may be too high. In order to reduce such computational complexity, we will introduce two approximate training methods in this section: negative sampling and hierarchical softmax. Since there is no major difference between the skip-gram model and the CBOW model, we will only use the skip-gram model as an example to introduce these two training methods in this section. 

- Subsampling: In text data, there are generally some words that appear at high frequencies, such “the”, “a”, and “in” in English. Generally speaking, in a context window, it is better to train the word embedding model when a word (such as “chip”) and a lower-frequency word (such as “microprocessor”) appear at the same time, rather than when a word appears with a higher-frequency word (such as “the”).

We use negative sampling for approximate training. For a central and context word pair, we randomly sample  𝐾  noise words ( 𝐾=5  in the experiment). According to the suggestion in the Word2vec paper, the noise word sampling probability  𝑃(𝑤)  is the ratio of the word frequency of  𝑤  to the total word frequency raised to the power of 0.75

In some cases, the cross-entropy loss function may have a disadvantage. GloVe uses squared loss and the word vector to fit global statistics computed in advance based on the entire dataset.

The central target word vector and context word vector of any word are equivalent in GloVe.

### Sub-word embeddings
English words usually have internal structures and formation methods. For example, we can deduce the relationship between “dog”, “dogs”, and “dogcatcher” by their spelling. All these words have the same root, “dog”, but they use different suffixes to change the meaning of the word. Moreover, this association can be extended to other words. For example, the relationship between “dog” and “dogs” is just like the relationship between “cat” and “cats”. The relationship between “boy” and “boyfriend” is just like the relationship between “girl” and “girlfriend”. 

#### fastText
In word2vec, we did not directly use morphology information. In both the skip-gram model and continuous bag-of-words model, we use different vectors to represent words with different forms. For example, “dog” and “dogs” are represented by two different vectors, while the relationship between these two vectors is not directly represented in the model. In view of this, fastText [Bojanowski et al., 2017] proposes the method of subword embedding, thereby attempting to introduce morphological information in the skip-gram model in word2vec.

In fastText, each central word is represented as a collection of subwords. Below we use the word “where” as an example to understand how subwords are formed. First, we add the special characters “<” and “>” at the beginning and end of the word to distinguish the subwords used as prefixes and suffixes. Then, we treat the word as a sequence of characters to extract the  𝑛 -grams. For example, when  𝑛=3 , we can get all subwords with a length of  3 :

`"<wh", "whe", "her", "ere", "re>", `

In fastText, for a word  𝑤 , we record the union of all its subwords with length of  3  to  6  and special subwords as  𝑤 . Thus, the dictionary is the union of the collection of subwords of all words. 

## BERT - Bidirectional Encoder Representations from Transformers
We have introduced several word embedding models for natural language understanding. After pretraining, the output can be thought of as a matrix where each row is a vector that represents a word of a predefined vocabulary. In fact, these word embedding models are all context-independent. Let us begin by illustrating this property.

Popular context-sensitive representations include TagLM (language-model-augmented sequence tagger) [Peters et al., 2017b], CoVe (Context Vectors) [McCann et al., 2017], and ELMo (Embeddings from Language Models) [Peters et al., 2018].

ELMo encodes context bidirectionally but uses task-specific architectures (however, it is practically non-trivial to craft a specific architecture for every natural language processing task); while GPT is task-agnostic but encodes context left-to-right.

BERT combines the best of both worlds: it encodes context bidirectionally and requires minimal architecture changes for a wide range of natural language processing tasks.

The embeddings of the BERT input sequence are the sum of the token embeddings, segment embeddings, and positional embeddings.

Pretraining BERT is composed of two tasks: masked language modeling and next sentence prediction. The former is able to encode bidirectional context for representing words, while the later explicitly models the logical relationship between text pairs.


