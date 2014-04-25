""" Filename: characterExtraction.py
Author: Emily Daniels
Date: April 2014
Purpose: Extracts character names from a text file and performs analysis of
text sentences containing the names.
Reference: See README.txt file for additional references """

import json
import nltk
import re
from collections import Counter
from collections import defaultdict
from nltk.corpus import stopwords
from pattern.en import parse, Sentence, mood
from pattern.db import csv
from pattern.vector import Document, NB

text = []
entityNames = []
chunkedSentences = ()
majorCharacters = []
sentenceList = []
characterSentences = defaultdict(list)
characterMoods = defaultdict(list)
characterTones = defaultdict(list)
sentenceAnalysis = defaultdict(list)


""" Reads the text from a text file. """


def readText(text):
    with open("730.txt", "rb") as f:
        text = f.read()
    return text

text = readText(text)


""" Parses text into parts of speech tagged with parts of speech labels.
Used for reference: https://gist.github.com/onyxfish/322906 """


def getChunkedSentences(chunkedSentences, text):
    sentences = nltk.sent_tokenize(text)
    tokenizedSentences = [nltk.word_tokenize(sentence)
                          for sentence in sentences]
    taggedSentences = [nltk.pos_tag(sentence)
                       for sentence in tokenizedSentences]
    chunkedSentences = nltk.batch_ne_chunk(taggedSentences, binary=True)
    return chunkedSentences

chunkedSentences = getChunkedSentences(chunkedSentences, text)


""" Creates a local list to hold nodes of tree passed through, extracting named
 entities from the chunked sentences.
Used for reference: https://gist.github.com/onyxfish/322906 """


def extractEntityNames(tree):
    entityNames = []
    try:
        if hasattr(tree, 'node') and tree.node:
            if tree.node == 'NE':
                entityNames.append(' '.join([child[0] for child in tree]))
            else:
                for child in tree:
                    entityNames.extend(extractEntityNames(child))
    except AttributeError:
        pass
    return entityNames


""" Uses the global entity list, creating a new dictionary with the properties
extended by the local list, without overwriting.
used for reference: https://gist.github.com/onyxfish/322906 """


def extendDict(chunkedSentences, entityNames):
    for tree in chunkedSentences:
        entityNames.extend(extractEntityNames(tree))
    return entityNames

entityNames = extendDict(chunkedSentences, entityNames)


""" Brings in stopwords and customstopwords to filter mismatches out. """


def removeStopwords(entityNames):
    with open("customStopWords.txt", "rb") as f:
        customStopwords = f.read().split(', ')
    for name in entityNames:
        if name in stopwords.words('english') or name in customStopwords:
            entityNames.remove(name)
    return entityNames

entityNames = removeStopwords(entityNames)


""" Adds names to the major character list if they appear frequently. """


def getMajorCharacters(entityNames, majorCharacters):
    for name in entityNames:
        if entityNames.count(name) > 10:
            majorCharacters.append(name)
    return majorCharacters

majorCharacters = list(set(getMajorCharacters(entityNames, majorCharacters)))


""" Split sentences on .?! "" and not on abbreviations of titles.
Used for reference: http://stackoverflow.com/a/8466725 """


def splitIntoSentences(text):
    sentenceEnders = re.compile(r"""
    # Split sentences on whitespace between them.
    (?:               # Group for two positive lookbehinds.
      (?<=[.!?])      # Either an end of sentence punct,
    | (?<=[.!?]['"])  # or end of sentence punct and quote.
    )                 # End group of two positive lookbehinds.
    (?<!  Mr\.   )    # Don't end sentence on "Mr."
    (?<!  Mrs\.  )    # Don't end sentence on "Mrs."
    (?<!  Ms\.  )     # Don't end sentence on "Ms."
    (?<!  Jr\.   )    # Don't end sentence on "Jr."
    (?<!  Dr\.   )    # Don't end sentence on "Dr."
    (?<!  Prof\. )    # Don't end sentence on "Prof."
    (?<!  Sr\.   )    # Don't end sentence on "Sr."
    \s+               # Split on whitespace between sentences.
    """, re.IGNORECASE | re.VERBOSE)
    return sentenceEnders.split(text)

sentenceList = splitIntoSentences(text)


""" Compares the list of sentences with the character names and returns
sentences that include names. """


def compareLists(sentenceList, majorCharacters, characterSentences):
    for sentence in sentenceList:
        for name in majorCharacters:
            if re.search(r"\b(?=\w)%s\b(?!\w)" % name, sentence,
                         re.IGNORECASE):
                characterSentences[name].append(sentence)
    return characterSentences

characterSentences = compareLists(sentenceList, majorCharacters,
                                  characterSentences)


""" Analyzes the sentence using grammatical mood module from pattern. """


def extractMood(characterSentences, characterMoods):
    for key, value in characterSentences.iteritems():
        for x in value:
            characterMoods[key].append(mood(Sentence
                                            (parse(str(x), lemmata=True))))
    return characterMoods

characterMoods = extractMood(characterSentences, characterMoods)


""" Trains a Naive Bayes classifier object with the reviews.csv file, analyzes
the sentence, and returns the tone. """


def extractSentiment(characterSentences, characterTones):
    nb = NB()
    for review, rating in csv("reviews.csv"):
        nb.train(Document(review, type=int(rating), stopwords=True))
    for key, value in characterSentences.iteritems():
        for x in value:
            characterTones[key].append(nb.classify(str(x)))
    return characterTones

characterTones = extractSentiment(characterSentences, characterTones)


""" Merges sentences, moods and tones together into one dictionary on each
 character. """


sentenceAnalysis = dict([(k, [characterSentences[k],
                              characterTones[k],
                              characterMoods[k]]) for k in characterSentences])


""" Writes the sentence analysis to a text file in the same directory. """


def writeAnalysis(sentenceAnalysis):
    with open("sentenceAnalysis.txt", "wb") as f:
        for item in sentenceAnalysis.items():
            f.write("%s:%s\n" % item)

writeAnalysis(sentenceAnalysis)


""" Writes the sentence analysis to a JSON file in the same directory. """


def writeToJSON(sentenceAnalysis):
    with open("sentenceAnalysis.json", "wb") as f:
        json.dump(sentenceAnalysis, f)

writeToJSON(sentenceAnalysis)
