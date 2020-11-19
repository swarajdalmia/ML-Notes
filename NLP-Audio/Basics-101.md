Book : Deep Learning for NLP and Speech Recognition by Uday Kamath, John Liu, Jimmy Whitaker 

# Definitions 
Highlight : The English language has about 170,000 words in its vocabulary but only about 10,000 are commonly used day-to-day. 

Highlight : Human communications have evolved to be highly efficient, allowing for reuse of shorter words whose meanings are resolved through context. This is
also present in speech example allophones. 

#### Text Analysis

**Morphology** : refers to the shape and internal structure of a word. 
**Lexical** : refers to the segmentation of text into meaningful units like words.
**Syntax** : rules and principles applied to combine words, phrases to form sentences. Or even sentences to form parapraphs ?
**Semantics** : relates to the meaning of words/sentences. It is semantics that provides the efficiency of human language. 
**Discourse** :  refers to conversations and the relationships that exist among sentences.
**Pragmatics** : refers to external characteristics such as the intent of the speaker to convey context

#### Speech Analysis
**Acoustics** : methods used to represent sounds 
**Phonemes** : basics units of speech. As characters are basic units of written language.
**Phonetics** : refers to how sounds are mapped to phonemes 
**Phonemics/Phonology** : refers to how phonemes are used in a language
**Prosodics** : refers to non-language characteris- tics that accompany speech such as tone, stress, intonation, and pitch

**Synchronic model** : a model that is based on a snapshot in time of a language. The complexity of diachronic models makes them difficult to handle, however, 
and synchronic models as originally championed by Swiss linguist Ferdinand de Saussure at the turn of the twentieth century are widely adopted today.
**Diachronic models** : models that can address changes in time.


## Morphological Analysis
All natural languages have systematic structure, even sign language. In linguistics, morphology is the study of the internal structure of words. 
Literally translated from its Greek roots, morphology means “the study of shape.” It refers to the set of rules and conventions used to form words based on their 
context, such as plurality, gender, contraction, and conjugation.

Words are composed of subcomponents called morphemes.
**Morphemes** : represent the smallest unit of language that hold independent meaning.

Some morphemes are words by themselves, such as “run”, “jump”. Other words are formed by a combination of morphemes, such as “runner”, “unhide.” 
Some languages, like English, have relatively simple morphologic rules for combining morphemes. Others, like Arabic, have a rich set of complex morphologic rules.
 
To humans, understanding the morphological relations between the words “walk, walking, walked” is relatively simple. The plurality of possible morpheme combinations,
however, makes it very difficult for computers to do so without morphological analysis. Two of the most common approaches are stemming and lemmatization, which 
we describe below.
 
### Stemming 
Often, the word ending is not as important as the root word itself. This is especially true of verbs, where the verb root may hold significantly more meaning than 
the verb tense. If this is the case, computational linguistics applies the process of word stemming to convert words to their root form.

works -> work 
worked -> work
workers -> work 

It is important to note that stemming can introduce ambiguity, as evident in the third example above where “workers” has the same stem as “works,” but both words 
have different meanings. On the other hand, the advantage of stemming is that it is generally robust to spelling errors, as the correct root may still be inferred 
correctly.

Note : One of the most popular stemming algorithms in NLP is the Porter stem- mer, devised by Martin Porter in 1980. This simple and efficient method uses a series 
of 5 steps to strip word suffixes and find word stems. Open-source implementations of the Porter stemmer are widely available.


### Lemmatization 
Lemmatization is another popular method used in computational linguistics to re- duce words to base forms. It is closely related to stemming in that it is an 
algorithmic process that removes inflection and suffixes to convert words into their lemma (i.e., dictionary form). 

works → works
worked → work 
workers → worker

Notice that the lemmatization results are very similar to those of stemming, except that the results are actual words. Whereas stemming is a process where 
meaning and context can be lost, lemmatization does a much better job as evident in the third example above. Since lemmatization requires a dictionary of lexicons
and numerous lookups, stemming is faster and the generally more preferred method. Lemmatization is also extremely sensitive to spelling errors, and may require
spell correction as a preprocessing step.

## Lexical Representations
Lexical analysis is the task of segmenting text into its lexical expressions. In the next few subsections, we provide an overview of word-level, sentence-level, 
and document-level representations. As the reader will see, these representations are inherently sparse, in that few elements are non-zero. We leave dense 
representations and word embeddings to a later chapter.

Insight : In natural language processing, often the first task is to segment text into separate words. Note that we say “often” and not “always.” As we will 
see later, sentence segmentation as a first step may provide some benefits, especially in the presence of “noisy” or ill-formed speech.

### Tokens 
The computational task of segmenting text into relevant words or units of meaning is called tokenization. Tokens may be words, numbers, or punctuation marks. 
In simplest form, tokenization can be achieved by splitting text using whitespace. This works in most cases, but fails in others. for example : "New York" should
be considered as a single token. To compound problems, tokens can sometimes consist of multiple words (e.g., “he who cannot be named”). 

There are also numerous languages that do not use any whitespace, such as Chinese.

Tokenization serves also to segment sentences by delineating the end of one sentence and beginning of another. Punctuation plays an important role in this task, 
but unfortunately punctuation is often ambiguous. Punctuation like apostrophes, hyphens, and periods can create problems. 

Example: 
Dr. Graham poured 0.5ml into the beaker.
|Dr.|, |Graham poured 0.|, |5ml into the beaker.|

Note : There are numerous methods to overcome this am- biguity, including augmenting punctuation-based with hand engineered rules, using regular expressions, 
machine learning classification, conditional random field, and slot-filling approaches.

### Stop Words
Tokens do not occur uniformly in English text. Instead, they follow an exponential occurrence pattern known as Zipf’s law, which states that a small subset of 
tokens occur very often (e.g., the, of, as) while most occur rarely. How rarely? Of the 884,000 tokens in Shakespeare’s complete works, 100 tokens comprise over
half of them. In the written English language, common functional words like “the,” “a,” or “is” provide little to no context, yet are often the most frequently 
occurring words in text as seen in Fig. 3.3. By excluding these words in natural language processing, performance can be significantly improved. 
The list of these commonly excluded words is known as a stop word list.

### N-Grams 
Work level representations just work in a bag of words model but provide no context of word ordering. They work with the Markov assumption that:
P(w1w2) = P(w1)\*P(w2)
This is also called a unigram model which considered the probabilities an independant from one another. 

Another approach is the ngram approach, where 

P(w1w2..wx) = Product[ P(wn|wn-1...w1) ]

### Document Representation 
1) Document Term matrix : just a matrix representation where the columns are unique words and the rows correspond to document ids. there are different ways of representing the value of each word in a document. Two are discussed below. 

a) Bag-of-words : The number at (i,j) is the count of word_i in document_j. This is also known as count vectorization. In practise a stop word filter is used before finding the bag of words model.
b) TFIDF : One of the issues in Bag of words is that rarer words get lot. TFIDF takes care of that by multiplying the Term Frequency with the Inverse Document Frequency. 

So w = (1+log(TF_t))\*Log(N/n_i))

TF_t = term frequency of term t in document d 
n_i = count of documents with term t 
N = total number of documents 

There are other formulas with small differences to account for different shortcomings but TFIDF is the most popular weighting method in document representation. 

## Syntactic Representations 

### Parts of Speech
A part-of-speech (POS) is a class of words with grammatical properties that play similar sentence syntax roles. It is widely accepted that there are 9 basic part of speech classes namely : nouns(dog), verbs(run), pronouns(you), adjectives(green), adverbs(quickly), article(the), preposition(for), conjunction(and), interjection(wow). 

There are numerous POS subclasses in English, such as singular nouns (NN), plural nouns (NNS), proper nouns (NP), or adverbial nouns (NR). Some languages can have over 1000 parts of speech. Due to the ambiguity of the English language, many English words belong to more than one part-of-speech category (e.g., “bank” can be a verb, noun, or interjection), and their role depends on how they are used within a sentence. There are two types of POS taggers : rule-based and statistical. 

Note : The Brown corpus was the first major collection of English texts used in computational linguistics research. It was created in the mid-1960s and consists of over a million words of English prose extracted from 500 randomly chosen publications of 2000 or more words. Each word in the corpus has been POS tagged meticulously using 87 distinct POS tags. The Brown corpus is still commonly used as a gold set to measure the performance of POS-tagging algorithms.

a) Rule Based : The best rules-based POS tagger to date achieved only 77% accuracy on the Brown corpus even with adhoc rules etc and fixes. 

b) Statistical : In the 1980's HMMs were introduced and they worked well. To account for more ambigu- ous word sequences, higher-order HMMs can also be used for larger sequences by leveraging the Viterbi algorithm. These higher-order HMMs can achieve very high accuracy, but they require significant computation load since they must explore a larger set of paths. Beyond HMMs, machine learning methods have gained huge popularity for POS tagging tasks, including CRF, SVM, perceptrons, and maximum entropy classification approaches. Most now achieve accuracy above 97%. In the subsequent chapters, we will examine deep learning approaches that hold even greater promise to POS tag prediction.

### Dependency Parsing 
In a natural language, grammar is the set of structural rules by which words and phrases are composed. Every sentence in English follows a certain pattern. Consider the fact that without grammar, there would be practically unlimited possibilities to combine words together. Most natural languages have a rich set of grammar rules, and knowl- edge of these rules helps us disambiguate context in a sentence. Grammar helps us disambiguate possible meanings, gives contexts to words, intents etc. 

Parsing is the natural language processing task of identifying the syntactic rela- tionship of words within a sentence, given the grammar rules of a language. 
There are two common ways of doing this : 

a) Constituent Grammar Parsing : representing the sentence by its constituent phrases, recursively down to the individual word level. Like a tree where the root contains the entire sentence and the leaves are the words. The first split is usally into (Noun Prase, Word Phrase).

b) Dependency Grammar : Link individual words by their dependency relationship. Maps a sentence to its dependency tree. The appeal of dependency tree is that the links closely resemble semantic relationships. Because a dependency tree contains one node per word, the parsing can be achieved with computational efficiency. 

Parsers are sub- divided into two general approaches. Top-down parsers use a recursive algorithm with a back-tracking mechanism to descend from the root down to all words in the sentence. Bottom-up parsers start with the words and build up the parse tree based on a shift/reduce or other algorithm. Top-down parsers will derive trees that will always be grammatically consistent, but may not align with all words in a sentence. Bottom-up approaches will align all words, but may not be always make grammati- cal sense.

### Context Free Grammars 
Because these rules are generally fixed and absolute, a context-free grammar (CFG) can be used to represent the grammatical rules of a language [JM09]. Context-free grammars typically have a representation known as Backus–Naur form. Unfortunately, because of the inherent ambiguity of language, CFG may generate multiple possible parse derivations for a given sentence. Probabilistic context-free grammars (PCFG) deal with this issue by ranking possible parse derivations and selecting the most probable. 

### Chunking 
For some applications, a full syntactic parse with its computational expense may not be needed. Chunking, also called shallow parsing, is a natural language pro- cessing task which joins words into base syntactic units rather than generating a full parse tree. 

### Treebanks 
A treebank is a text corpus that has been parsed and annotated for syntactic structure. That is, each sentence in the corpus has been parsed into its dependency parse tree. Treebanks are typically generated iteratively using a parser algorithm and human review. The creation of treebanks revolu- tionized computational linguistics, as it embodied a data-driven approach to gener- ating grammars that could be reused broadly in multiple applications and domains. Statistical parsers trained with treebanks are able to deal much better with structural ambiguities. 

Note : **The Penn Treebank** is the de facto standard treebank for parse analysis and evaluation. Initially released in 1992, it consists of a collection of articles from Dow Jones News Service written in English, of which 1 million words are POS-tagged and 1.6 million words parsed with a tagset. An improved version of the Penn Treebank was released in 1995.

Note : **Universal Dependencies** is a collection of over 100 treebanks in 60 languages, created with the goal of facilitating cross-lingual analysis [McD+13]. As its name implies, the treebanks are created with a set of universal, cross-linguistically consistent grammatical annotations. The first version was released in October of 2014.

## Semantic Representations 
Whereas lexical and syntactic analyses capture the form and order of language, they do not associate meaning with words or phrases.


### Named Entity Recognition :
Named entity recognition (NER) is a task in natural language processing that seeks to identify and label words or phrases in text that refer to a person, location, or- ganization, date, time, or quantity. Ambiguities can exist in two ways: different entities of the same type (George Washington and Wash- ington Carver are both persons) or entities of different types (George Washington or Washington state). While regular expressions can be used to some extent for name entity recognition, the standard approach is to treat it as a sequence labeling task or HMM in similar fashion to POS-tagging or chunking. Conditional random fields (CRFs) have shown some success in named entity recognition. However, training a CRF model typically requires a large corpus of annotated training data. Even with a lot of data, name entity recognition is still largely unsolved.

### Relation Extraction
Relationship extraction is the task of detecting semantic relationships of named en- tity mentions in text. The common approach to relation extraction is to divide it into subtasks:
1. Identify any relations between entities 
2. Classify the identified relations by type 
3. Derive logical/reciprocal relations.

The first subtask is typically treated as a classification problem, where a binary decision is made as to whether a relation is present between any two entities within the text. The second subtask is a multiclass prediction problem. Naive Bayes and SVM models have been successfully applied to both subtasks.

### Event extraction 

Events are mentions within text that have a specific location and instance or interval in time associated with them. Some examples of events are: the Superbowl, The Cherry Blossom festival, and our 25th wedding an- niversary celebration. Both rules-based and machine learning approaches for event detection are similar to those for relationship extraction. Such approaches have had mixed success due to the need for external context and the importance of temporal relations.

### Semantic Role Labelling 

Semantic role labeling (SRL), also known as thematic role labeling or shallow se- mantic parsing, is the process of assigning labels to words and phrases that indicate their semantic role in the sentence. A semantic role is an abstract linguistic construct that refers to the role that a subject or object takes on with respect to a verb. These roles include: agent, experiencer, theme, patient, instrument, recipient, source, ben- eficiary, manner, goal, or result.

Note : **PropBank (the Proposition Bank)** is a corpus of Penn Treebank sentences fully annotated with semantic roles, where each of the roles is specific to an individual verb sense. Each verb maps to a single instance in PropBank. The corpus was released in 2005.

Note : **FrameNet** is another corpus of sentences annotated with semantic roles. Whereas PropBank roles are specific to individual verbs, FrameNet roles are specific to semantic frames. A frame is the background or setting in which a semantic role takes place—it provides a rich set of contexts for the roles within the frame. FrameNet roles have much finer grain than those of PropBank. FrameNet contains over 1200 semantic frames, 13,000 lexical units, and 202,000 example sentences.

## Discourse Representation 
Discourse analysis is the study of the structure, relations, and meaning in units of text that are longer than a single sentence. It encompasses characteristics such as the document/dialogue structure, topics of discussion, cohesion, and coherence of the text. Two popular tasks in discourse analysis are coreference resolution and discourse segmentation.

### Cohesion
Cohesion is a measure of the structure and dependencies of sentences within dis- course. It is defined as the presence of information elsewhere in the text that supports presuppositions within the text. That is, cohesion provides continuity in word and sentence structure. It is sometimes called “surface level” text unity, since it provides the means to link structurally unrelated phrases and sentences together. There are six types of cohesion within text: coreference, substitution, ellipsis, conjunction, reiteration, and collocations.

example :
Jack ran up the hill. He walked back down. (jack and he provide cohesion)

### Coherence 

Coherence refers to the existence of semantic meaning to tie phrases and sentences together within text. It can be defined as continuity in meaning and context, and usually requires inference and real-world knowledge. 

example: 
Jack carried the bucket. He spilled the water. (bucket and water)

## Language Model 

It predicts the probabibility of the current word given the occurance of words in the past. For instance, language models can be used for spell correction by predicting a word w_i given all of the previous words before i. It typically uses an n-gram model. We assume that the probability of observing the ith word wi in the context history of the preceding words can be approximated by the probability of observing it in the shortened context history of the preceding words (nth order Markov property). 

### Laplace Smoothing 
The sparsity of n-grams can become a problem, especially if the set of documents used to create the n-grams language model is small. In those cases, it is not uncom- mon for certain n-grams to have zero counts in the data. The language model would assign zero probability to these n-grams. This creates a problem when these n-grams occur in test data. Because of the Markov assumption, the probability of a sequence is equal to the product of the individual probabilities of the n-grams. A single zero probability n-gram would set the probability of the sequence to be zero.

The simplest smoothing algorithm initializes the count of every possible n-gram at 1. We add 1 to every count so it’s never zero. To balance this, we add the number of possible words to the divisor, so the division will never be greater than 1.

A more effective and wisely used method is Kneser–Ney smoothing, due to its use of absolute discounting by subtracting a fixed value from the probability’s lower order terms to omit n-grams with lower frequencies. 

#### Out of Vocab 
Another serious problem for language models arise when the word is not in the vo- cabulary of the model itself. Out-of-vocabulary (OOV) words create serious prob- lems for language models. In such a scenario, the n-grams that contain an out-of- vocabulary word are ignored. The n-gram probabilities are smoothed over all the words in the vocabulary even if they were not observed. 

To explicitly model the probability of out-of-vocabulary words, we can intro- duce a special token (e.g., <unk>) into the vocabulary. Out-of-vocabulary words in the corpus are effectively replaced with this special <unk> token before n-grams counts are accumulated. With this option, it is possible to estimate the transition probabilities of n-grams involving out-of-vocabulary words. By doing so, however, we treat all OOV words as a single entity, ignoring the linguistic information
 
 Another approach is to use approximate n-gram matching. OOV n-grams are mapped to the closest n-gram that exists in the vocabulary, where proximity is based on some semantic measure of closeness (we will describe word embeddings in more detail in a later chapter).
 
 A simpler way to deal with OOV n-grams is the practice of backoff, based on the concept of counting smaller n-grams with OOV terms. If no trigram is found, we instead count bigrams. If no bigram found, use unigrams.
 
 ### Perplexity 
 The most commonly used method for measuring language model performance is perplexity. Perplexity can be thought of as how surprised a model is by a sequence of words at test time. More technically, the perplexity of a language model is equal to the geometric average of the inverse probability of the words measured on test data. Note that it is impor- tant for the test sequence to be comprised of the same n-grams as was used to train the language model, or else the perplexity will be very high.

## Text Classification 
Text classification is a core task in many applications such as information retrieval, spam detection, or sentiment analysis. The goal of text classification is to assign doc- uments to one or more categories. The most common approach to building classi- fiers is through supervised machine learning whereby classification rules are learned from examples. 

### Emotional State Models 
An emotional state model is one that captures the human states of emotion. The Mehrabian and Russell model, for instance, decomposes human emotional states into three dimensions:

- Valence : Measures the pleasurableness of an emotion also known as polarity. Ambivalence is the conflict between positive and negative valence.
- Asousal : Measures the intensity of emotion.
- Dominance : Measures the dominion of an emotion over others.

There are other emotional state models used in sentiment analysis, including Plutchik’s wheel of emotions and Russell’s two-dimensional emotion circumplex model.

The simplest computational approach to sentiment analysis is to take the set of words that describe emotional states and vectorize them with the dimensional values of the emotional state model. The occurrence of these words is computed within a document, and the sentiment of the document is equal to the aggregated scores of the words. This lexical approach is very fast, but suffers from the inability to effectively model subtlety, sarcasm, or metaphor.  Negation (e.g., “not nice” vs. “nice”) is also problematic with pure lexical approaches.

Note : The affective norms for English words (ANEW) dataset is a lexicon created by Bradley and Lang containing 1000 words scored for emotional ratings of valence, dominance, and arousal. ANEW is very useful for longer texts and newswire documents. Another model is the SentiStrength model for short informal text developed by Thelwall et al., which has been applied successfully to analyze text and Twitter messages.

### Subjectivity and Objectivity Detection

A closely related task in sentiment analysis is to grade the subjectivity or objec- tivity of a particular piece of text. Objectivity detection could help identify personal bias, track hidden viewpoints, and alleviate the “fake news” problem existing today. 

### Entailment 
Textual entailment is the logical concept that truth in one text fragment leads to truth in another text fragment. Entailment is considered a text classification problem. It has widespread use in many NLP applications (e.g., question answering). Initial approaches toward entailment were logical-form based methods that required a many axioms, inference rules, and a large knowledge base. These theorem-proving methods performed poorly in comparison to other statistical NLP approaches. Entailment remains an open research topic.

## Text Clustering 
While text classification is the usual go-to approach for text analytics, we are often presented with a large corpus of unlabeled data in which we seek to find texts that share common language and/or meaning.The most common approach to text clustering is via the k-means algorithm. Text documents are tokenized, sometimes stemmed or lemmatized, stop words are removed, and text is vectorized using bag-of-words or TFIDF. K-means is applied to the resulting document-term matrix for different k values. The first is the notion of distance between two text fragments. For k-means, this is the Euclidean distance, but other measures like cosine distance could theoretically be used. The second is determining the value of k—how many different clusters of text exist within a cor- pus. As in standard k-means, the elbow method is most widely used for determining the value of k.

#### LSA

Latent semantic analysis (LSA) is a technique that seeks to identify relationships between a set of documents and words based on the implicit belief that words close in meaning will occur in similar pieces of text. It is one of the oldest methods for topic modeling. It uses a mathematical technique named singular value decomposition (SVD) to convert the document-term matrix of a text corpus into two lower-rank matrices: a document-topic matrix that maps topics to documents, and a topic-word matrix that maps words to topics.

#### LDA 

Latent Dirichlet allocation (LDA) is a model that also acts to decompose a document-term matrix into a lower-order document-topic matrix and topic-word matrix. It differs from LSA in that it takes a stochastic, generative model approach and assumes topics to have a sparse Dirichlet prior. 

## Machine Translation

Language translation is hard even for humans to be able to fully capture meaning, tone, and style. Languages can have significantly different morphology, syntax, or semantic structure. For instance, it will be rare to find English words with more than 4 morphemes, but it is quite common in Turkish or Arabic. German sentences commonly follow the subject- verb-object syntactic structure, while Japanese mostly follows a subject-object-verb order, and Arabic prefers a verb-subject-object order. With machine translation, we typically focus on two measures:

- Faithfulness = preserving the meaning of text in translation
- Fluency = natural sounding text or speech to a native speaker.

### Dictionary Based 

In simplest form, machine translation can be achieved by a direct translation of each word using a bilingual dictionary. A slight improvement may be to directly translate word phrases instead of individual words. Because of the lack of syntactic or semantic context, direct translation tends to do poorly in all but the simplest machine translation tasks. 

Another classical method for machine translation is based on learning lexical and syntactic transfer rules from the source to the target language. These rules provide a means to map the parse trees between languages, potentially altering the structure in the transformation

### Statistical Translation 
Statistical machine translation adopts a probabilistic approach to map from one lan- guage to another. Specifically, it builds two types of models by treating the problem as one similar to a Bayesian noisy channel problem in communications:
- Language model (fluency) = P(X)
- Translation model (faithfulness) = P(Y|X).

It tries to maximise for the product of both. Statistical models are based on the notion of word alignment, which is a mapping of a sequence of words from the source language to those of a target language. Because of differences between languages, this mapping will almost never be one- to-one. 

Note : BLEU (bilingual evaluation understudy) is a common method to measure the quality of machine translation [Pap+02]. It measures the similarity be- tween phrase-based model translations and human-created translations aver- aged over an entire corpus. Similar to precision, it is normally expressed as a value between 0 and 1 but sometimes scaled by a factor of 10.

## Question Answering 
Question answering (QA) is the NLP task of answering questions in natural lan- guage. It can leverage expert system, knowledge representation, and information retrieval methods. Traditionally, question answering is a multi-step process where relevant documents are retrieved, useful information is extracted from these docu- ments, possible answers are proposed and scored against evidence, and a short text answer in natural language is generated as a response.

Early question answering systems focused only on answering a predefined set of topics within a particular domain [KM11]. These were known as closed-domain QA systems, as opposed to open-domain QA systems that attempt to answer queries in any topic. 

Question decomposition is the first step in any QA system, where a question is processed to form a query. In simple versions, questions would be parsed to find keywords which served as queries to an expert system to produce answers. This is known as query formation, where keywords are extracted from the question to for- mulate a relevant query. Another method is query reformation, where the entities in the question are extracted along with its semantic relation. For instance, the following sentence and semantic relation:

Who invented the telegraph? → Invented (Person, telegraph)


### Information Retreival Based 
Web-based question answering systems like Google Search are based on informa- tion retrieval (IR) methods that leverage the web. These text-based systems seek to answer questions by finding short texts from the internet or some other large collec- tion of documents. Typically, they map queries into a bag-of-words and use methods like LSA to retrieve a set of relevant documents and extract passages within them. Depending on the question type, answer strings can be generated with a pattern- extraction approach or n-gram tiling methods. 

### Knowledge-Based QA
Knowledge-based question answering systems, on the other hand, take a semantic approach. They apply semantic parsing to map questions into relational queries over a comprehensive database. This database can be a relational database or knowledge base of relational triples (e.g., subject-predicate-object) capturing real-world relationships such as DBpedia or Freebas. Because of their ability to capture meaning, knowledge-based methods are more applicable for advanced, open-domain question-answering applications as they can bring in external information in the form of knowledge bases. At the same time, they are constrained by the set relations of those knowledge bases.

Note : DBpedia is a free semantic relation database with 4.6 million entities ex- tracted from Wikipedia pages in multiple languages. It contains over 3 billion relational triples expressed in the resource description framework (RDF) format. DBpedia is often considered the foundation for the semantic web, also known as the linked open data cloud. First released in 2007, DBpedia continues to evolve through crowdsourced updates in similar fashion to Wikipedia.

### Automated Reasoning 
By creating a set of first-order logic clauses, QA systems can enhance a set of semantic relations and evidence retrieved in support of answer hypotheses.

A common metric used to measure question answering system performance is mean reciprocal rank (MRR). It is based on using a gold set of questions that have been manually labeled by humans with correct answers. To evaluate a QA system, the set of ranked answers of the system would be compared with the gold set labels of a corpus of N questions.

## Automatic Summarization 
Automatic summarization is a useful NLP task that identifies the most relevant in- formation in a document or group of documents and creates a summary of the con- tent. It can be an extraction task that takes the most relevant phrases or sentences in original form and uses them to generate the summary, or an abstraction task that generates natural language summaries from the semantic content. Both approaches mirror how humans tend to summarize text, though the former extracts text while the latter paraphrases text.

### Extraction Based
In most implementations, it simply extracts a subset of sentences deemed most important. One method to measure importance is to count informative words based on lexical measures (e.g., TFIDF).  Another is to use discourse measures (e.g., coherence) to identify key sentences. Centroid-based methods evaluate word probability relative to the background corpus to determine importance. A creative approach called TextRank takes a graph-based approach to assign sentence scores based on lexical similarity of words.

### Abstraction Based 
Unlike extraction-based copying, abstraction-based approaches take a semantic approach. One method is to use entity recognition and semantic role labeling to identify relations. 
