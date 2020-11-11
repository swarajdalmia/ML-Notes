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

Tokenization serves also to segment sentences by delineating the end of one sen- tence and beginning of another. Punctuation plays an important role in this task, 
but unfortunately punctuation is often ambiguous. Punctuation like apostrophes, hy- phens, and periods can create problems. 

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












