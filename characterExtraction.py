# Filename: characterExtraction.py
# Author: Emily Daniels
# Date: April 2014
# Purpose: Extracts character names from a text file and performs analysis of text sentences containing the names.
# Reference: See README.txt file for additional references

import nltk #Natural Language Toolkit
from collections import Counter #used to count names
import re #regular expression operations
from collections import defaultdict #returns a new dictionary-like object
import json #JavaScript Object Notation

text = [] #holds the list of strings read from the text file
entityNames = [] #holds the entity names
chunkedSentences = () #holds the tagged and chunked sentences 
majorCharacters = [] #holds the list of major characters
sentenceList = [] #holds the list of extracted sentences
characterSentences = defaultdict(list) #holds the dictionary of character sentences
characterMoods = defaultdict(list) #holds the dictionary of character moods
characterTones = defaultdict(list) #holds the dictionary of character tones
sentenceAnalysis = defaultdict(list) #holds the dictionary of sentence analysis

#reads the text from a text file
def readText(text):
    file = open("730.txt", "rb")
    text = file.read()
    file.close()
    return text

text = readText(text)

#parses text into parts of speech tagged with pos labels
#used for reference: https://gist.github.com/onyxfish/322906
def getChunkedSentences(chunkedSentences, text):
    sentences = nltk.sent_tokenize(text)
    tokenizedSentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    taggedSentences = [nltk.pos_tag(sentence) for sentence in tokenizedSentences]
    chunkedSentences = nltk.batch_ne_chunk(taggedSentences, binary=True)
    return chunkedSentences

chunkedSentences = getChunkedSentences(chunkedSentences, text)

#creates a local list to hold nodes of tree passed through, extracting named entities from the chunked sentences
#used for reference: https://gist.github.com/onyxfish/322906
def extractEntityNames(tree):
    entityNames = []
    if hasattr(tree, 'node') and tree.node:
        if tree.node == 'NE':
            entityNames.append(' '.join([child[0] for child in tree]))
        else:
            for child in tree:
                entityNames.extend(extractEntityNames(child))
    return entityNames

#uses the global entity list, creating a new dictionary with the properties extended by the local list, without overwriting
#used for reference: https://gist.github.com/onyxfish/322906
def extendDict(chunkedSentences, entityNames):
    for tree in chunkedSentences:
        entityNames.extend(extractEntityNames(tree))
    return entityNames

entityNames = extendDict(chunkedSentences, entityNames)
print ("Done writing entity names.")

#brings in stopwords and customstopwords to filter mismatches out
def removeStopwords(entityNames):
    customStopwords = []
    file = open("customStopWords.txt", "rb")
    customStopwords = file.read().split(', ')
    file.close()
    from nltk.corpus import stopwords
    for name in entityNames:
        if name in stopwords.words('english') or name in customStopwords:
            entityNames.remove(name)
    return entityNames

entityNames = removeStopwords(entityNames)

#adds names to the major character list if they appear frequently
def getMajorCharacters(entityNames, majorCharacters):
    for name in entityNames:
        if entityNames.count(name) > 10:
            majorCharacters.append(name)
    return majorCharacters

majorCharacters = getMajorCharacters(entityNames, majorCharacters)

#removes duplicates
majorCharacters = list(set(majorCharacters))
#print majorCharacters
print ("Done writing major characters.")

#split sentences on .?! "" and not on abbreviations of titles
#used for reference: http://stackoverflow.com/a/8466725
def splitIntoSentences(text, sentenceList):
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
    """, 
    re.IGNORECASE | re.VERBOSE)
    sentenceList = sentenceEnders.split(text)
    return sentenceList

sentenceList = splitIntoSentences(text, sentenceList)

#compares the list of sentences with the character names and returns sentences that include names
def compareLists(sentenceList, majorCharacters, characterSentences):
    for sentence in sentenceList:
        for name in majorCharacters:
            if re.search(r"\b(?=\w)%s\b(?!\w)" % name, sentence, re.IGNORECASE):
                characterSentences[name].append(sentence)          
    return characterSentences

characterSentences = compareLists(sentenceList, majorCharacters, characterSentences)

#imports mood from pattern library, analyzes the sentence, and returns the grammatical mood
def extractMood(characterSentences, characterMoods):
    from pattern.en import parse, Sentence
    from pattern.en import mood
    for key,value in characterSentences.iteritems():
        for x in value:
            x = parse(str(x), lemmata=True)
            x = Sentence(x)
            characterMoods[key].append(mood(x))
    return characterMoods

characterMoods = extractMood(characterSentences, characterMoods)

#trains a Naive Bayes classifier object with the reviews.csv file, analyzes the sentence, and returns the tone
def extractSentiment(characterSentences, characterTones):
    from pattern.vector import Document, NB
    from pattern.db import csv
    nb = NB()
    for review, rating in csv("reviews.csv"):
        v = Document(review, type=int(rating), stopwords=True)
        nb.train(v) 
    for key, value in characterSentences.iteritems():
        for x in value:
            characterTones[key].append(nb.classify(str(x)))
    return characterTones
 
characterTones = extractSentiment(characterSentences, characterTones)

#merges sentences, moods and tones together into one dictionary on each character
sentenceAnalysis = dict([(k,[characterSentences[k],characterTones[k],characterMoods[k]]) for k in characterSentences])

#writes the sentence analysis into a text file
def writeAnalysis(sentenceAnalysis):            
    file = open("sentenceAnalysis.txt", "wb")
    for item in sentenceAnalysis.items():
        file.write("%s:%s\n" % item)
    file.close()
    print ("Done writing sentenceAnalysis.")

writeAnalysis(sentenceAnalysis)

#writes the combined dictionaries to a JSON file
def writeToJSON(sentenceAnalysis):    
    with open("sentenceAnalysis.json", "wb") as outfile:
        json.dump(sentenceAnalysis, outfile)
    print ("Done writing to JSON.")

writeToJSON(sentenceAnalysis)
