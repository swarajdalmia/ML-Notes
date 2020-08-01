# Stanford CS224N: NLP with Deep Learning | Winter 2019

[Youtube course](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z)
[Course Website](https://web.stanford.edu/class/cs224n/index.html#schedule)

# Lec-1: Introduction and Word Vectors

- Are we very difference from oranutans in terms of intelligence ?
- Our knowledge of language or the language machine in the brain is much less older than the vision system. 
- bandwidth of human communication(few words per sec) vs those of computers (100mbs). One of the interesting things about lanaguge is that it compresses information, for example, a visual scene can be described with just a couple sentences. Normally to convey that task one would require a lot more information. 

## How do we represent the meanings of words ?
- words have rich meanings 
- what is meaning : the idea or thing that the symbol represents. 

How to represent meaning in computers: 
- WordNet : a thesaurus containing list of synonyms and hypernyms. For every word gives example of senses of the word meaning when its a noun, verb, adverb, adjective. 
- The problem with wordnet: misses nuance, it is incompete and misses new meanings of words, it can't commpute similarity between words. 

 Words can be represented as one hot vectors
- however there are too many words in a language and it doesn't consider relationship between words. How to represent words like run and running. Words derived from the same root. All vectors are orthogonal and no similarity. 
- could we not use word similarity tables ? Google tried that in 2005. However, the matrix to represent that would be huge. 
- instead, lets use vectors to encode similarity between words. 

Distributional Semantics : A words meaning is given by the words that frequently appear close by. "you shall know a word by the company it keeps" - one of the most successful ideas of modern NLP. 
- use the many contexts of w to build a representation of w
- this differs, in terms of how humans understand language ? to some extent it is similar. 

Word Vectors: They are dense vectors(with most elements non zero) such that it captures similarity between different word vectors. The minimum dimensionality ppl use is 50, for ones laptop 300 is a good starting point and for performance 3-5k maybe. 

Word vectors are also called word embeddings and they are a distributed representation. 

Visualization - looked at projections of 100 dim vectors. 

## Word2vec

Word2vec(introduced in [mikolov et al. 2013](https://arxiv.org/abs/1301.3781)) is a framework for learning word vectors. 

The idea:
- have a large corpus so as to learn the various contexts meanings of the words 
- every word in the vocab will be represented by a vector 
- every position in the text, has a centre word c and context words around it, o. 
- the similarity of the word vecotrs for c and o are used to calculate the probability of o given c or vice-versa. 
- the vectors are kept on adjusted to maximise the probability of prediction 

There are no common embeddings used across NLP. There is a learning algorithm, to learn these word embeddings. The dimensions of the vecotr space are arbitrary and one shouldn't read too much into the meaning of the individual elements. There is some meaning at times though. What matters more is how close the words are in the vector space. 

Emergence of word2vec was one of the biggest things that changed the direction of the NLP field. It is a simple and scalable way of learning word vectors.  

![](../images/word2vec_likelihood.jpeg) 

The above captures the likelihood first over each of the centre words and for each centre word, around the context window. 

But how to arrive at `P(w_t+j | w_t ; theta) i.e. probability of a context word given a centre word and the parameters theta. 

For each word w, there are two vectors associated with it :
- v_w when w is a centre word 
- u_w when w is a context word 

Then, for a centre word c and a context word o:
p(o|c) = exp(transpose(u_o) * v_c)/sum over all words w(exp(u_w * v_c))

Exponentiation makes everything positive. The numerator is the dot product, and the denominator normalises ovee the entire vocab. The above is similar to the suftmax function. Ensures, summation is = 1. 

The number of parameters in this model is(given d dimensional vectors and V = vocab) = 2*V*d since every word has 2 vectors. 

How to arrive at the u and v vectors ? We start with random values and then iteratively change them to minimise the loss function. 

Note: we are using one probability distribution to predict all the words in the context. Therefore the probability of the context words is usually quite low. However we still expect it to capture relative probabilities as precisely as possible. 

There are two popular word2vec representations : skipgram and continuous bag of words. One also requires different training methods, for example negative sampling. In the usual method the noralisation step is too expensive so we use negative sampling. The idea is to train a binary regression for a true pair vs that of a noise pair. 

# Lec-2 : Word Vectors and Word Senses

Words like "the", "a", "and" occur frequently so will they have a high probability always ? and how to deal with them ? 

Why are 2 vectors used for word vecs ? it ensures easy optimisation. Both are avergaed at the end ! But it is also possible to do with it just one vector per word. 

The optimisation involed computers functions of all windows in the corpus(possibly billions) and it is very expensive to compute. This is a bad idea and practially stochastic gradient descent is used which repeatedly samples random windows and updates after each one. 

Another interesting thing compared to computer vision is that in each optimisation step the gradient update matrix is very sparce. Since the gradients are taken at each window which has 2m+1 words at most. To ensure a faster update incase of sparce matixes, one can use hashes for word vectors to access them faster. 

So far for training the word vectors, we used the naive softmax method. It is simple but has an expensive training method. So instead negative sampling is used. In the naive softmax, to calculate the p(o|c) the normalisation term is too computationally expensive and instead, binary logistic regressions are trained for true pairs vs several noise pairs(randomly sampled). In the paper that discussed negative sampling, the probability of two words cooccuring is maximised. For each true word, k randomly chosen negaitves are used. The probability of the real word is maximised and the prob for the random co-occurances are minimised. 

To deal with the problem of very frequent words, the unigram distribution to the power 3/4 is used. this makes frequent words, sample less often and non frequent words sample more often. 

### Why not capture co-occurance counts directly ?

To calculate these probabilities P(centre word | context word) and vice-versa we essentially need cooccurance probbilities. So why dont we use the co-occurance matrix. The method discussed below was used before 2013. Use a wondow based co-occurance matrix and use simple counts.

However there are many issues with these:
- inc in size with inc in vocab 
- very high dimensional matrix which is sparce

What one did was use dimensioanlity reduction to get a low dimensional vector similar to word2vec. Use singular value decomposition(SVD). Explains SVD. A consise and simple explanation. Discusses issues with SVD and hacks to make it better.

### GloVe

 
