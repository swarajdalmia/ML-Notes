Ch - 8 from the book "Deep Learning for NLP and Speech Recognition" by Uday Kamath, John Liu, Jimmy Whitaker

# Automatic Speech Recognition 

This chapter introduces the fundamental concepts of speech recognition with a focus on HMM-based methods.
The focus of ASR is to convert a digitized speech signal into computer readable text, referred to as the transcript.

These models must be robust to variations in speakers, acoustic environments, and context. For example, human speech can have any combination of time variation 
(speaker speed), articulation, pronunciation, speaker volume, and vocal variations (raspy or nasally speech) and still result in the same transcript.

Linguistically, additional variables are encountered such as prosody (rising in- tonation when asking a question), mannerisms, spontaneous speech, also known as 
filler words (“um”s or “uh”s), all can imply different emotions or implications, even though the same words are spoken. Combining these variables with any number
of environmental scenarios such as audio quality, microphone distance, background noise, reverberation, and echoes exponentially increases the complexity of the 
recognition task.

The topic of speech recognition can include many tasks such as keyword spotting, voice commands, and speaker verification (security). In the interest of concision, 
we focus mainly on the task of speech-to-text (STT), specifically, large vocabulary continuous speech recognition (LVCSR) in this chapter.

The classical approach to ASR (non-deep) is discussed below. 

## Acoustic Features
The selection of acoustic features for ASR is a crucial step. Features extracted from the acoustic signal are the fundamental components for any model building as
well as the most informative component for the artifacts in the acoustic signal. Thus, the acoustic features must be descriptive enough to provide useful 
information about the signal, as well as resilient enough to the many perturbations that can arise in the acoustic environment.
