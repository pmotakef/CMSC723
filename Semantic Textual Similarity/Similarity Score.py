import math
import sys
import subprocess
import nltk
import collections
import json


main_paragram_file = "paragram-phrase-XXL.txt"
backtrack_paragram_file = "paragram_300_sl999.txt"
minimal_paragram_file = "paragram_vectors.txt"


class ParagramLoader:
    def __init__(self, vocab, embedingFilename = "paragram-phrase-XXL.txt", largeEmbeddingfilename = "paragram_300_sl999.txt"):
        self.wordEmbedding = {}
        self.largeParagramEmbeding = {}
        self.loadedInMemory = False
        self.embedFile = embedingFilename
        self.largeEmbedFile = largeEmbeddingfilename
        self.vocabulary = vocab

    def loadParagramEmbeddingInMemory(self):
        print "Started Loading " + self.embedFile
        with open(self.embedFile, "rb") as f:
            for line in f:
                embedArray = []
                if (not line):  # Line doesn't hold any data
                    continue
                if (self.embedFile == minimal_paragram_file):
                    embed = line.replace('\n', '').split('\t')
                else:
                    embed = line.replace('\n', '').split(' ')
                for i in range(1, len(embed)):
                    embedArray += [float(embed[i])]
                self.wordEmbedding[embed[0]] = embedArray
        self.loadedInMemory = True
        print "Finished loading " + self.embedFile

    def loadBacktrackParagramEmbedding(self):
        print "Started loading " + self.largeEmbedFile
        self.readFilteredWordsFromBigParagramFile(self.filterBacktrackWords())
        print "Finished loading " + self.largeEmbedFile

    def findWordEmbeddingFromParagram(self, word):
        if (self.loadedInMemory):
            if (word in self.wordEmbedding):
                return self.wordEmbedding[word]
            if (self.largeParagramEmbeding and word in self.largeParagramEmbeding):
                return self.largeParagramEmbeding[word]
            if (word.lower() in self.wordEmbedding):
                return self.wordEmbedding[word.lower()]
            if (self.largeParagramEmbeding and word.lower() in self.largeParagramEmbeding):
                return self.largeParagramEmbeding[word.lower()]
            return []


    # Calculates the paragram embeddings for the given sentence
    def calcPragramEmbedding(self, sentence):
        embedArraySent = []
        firstUpdate = True
        updateCounter = 0.0
        for word in sentence:
            correctWord = word
            if (not word):
                continue
            embedArrayForWord = self.findWordEmbeddingFromParagram(correctWord)
            updateCounter += 1.0

            if (len(embedArrayForWord) < 25):
                continue

            if (firstUpdate):
                firstUpdate = False
                for i in range(0, len(embedArrayForWord)):
                    embedArraySent += [embedArrayForWord[i]]
            else:
                for i in range(0, len(embedArrayForWord)):
                    embedArraySent[i] += embedArrayForWord[i]


        for i in range(0, len(embedArraySent)):
            embedArraySent[i] /= updateCounter
        return (embedArraySent)

    def calcCosineSimilarity(self, sentence1, sentence2):
        if (self.loadedInMemory):
            embedParagramSent1 = self.calcPragramEmbedding(sentence1)
            embedParagramSent2 = self.calcPragramEmbedding(sentence2)
        else:
            return 0.0

        if (len(embedParagramSent1) != len(embedParagramSent2)):
            return 0.0

        #Compute dot product
        dotprod = 0.0
        s1 = 0.0
        s2 = 0.0
        for i in range(0, len(embedParagramSent1)):
            dotprod += (embedParagramSent1[i] * embedParagramSent2[i])
            s1 += (embedParagramSent1[i] * embedParagramSent1[i])
            s2 += (embedParagramSent2[i] * embedParagramSent2[i])

        if (abs(dotprod) < 0.000000001 ):
            return 0.0

        s1 = math.sqrt(s1)
        s2 = math.sqrt(s2)

        return (dotprod / (s1 * s2))

    def release(self):
        self.wordEmbedding.clear()
        self.largeParagramEmbeding.clear()


    def filterBacktrackWords(self):
        if (not self.loadedInMemory):
            return
        filteredWords = []
        for word in self.vocabulary:
            if (not word in self.wordEmbedding and not word.lower() in self.wordEmbedding):
                filteredWords += [word]
        return filteredWords

    def readFilteredWordsFromBigParagramFile(self, words):
        if words == []:
            return

        wcopy = []
        for w in words:
            wcopy += [w.lower()]

        print "Total of " + str(len(wcopy)) + " words require backtrack"

        with open(self.largeEmbedFile, "rb") as f:
            self.largeParagramEmbeding = {}
            for line in f:
                embedArray = []
                if (not line):  # Line doesn't hold any data
                    print "Error reading line"
                    continue
                if (self.embedFile == minimal_paragram_file):
                    embed = line.replace('\n', '').split('\t')
                else:
                    embed = line.replace('\n', '').split(' ')
                try:
                    curemb = embed[0].decode('utf-8')

                except:
                    print "Couldn't decode " + embed[0]
                    continue

                if (curemb.lower() in wcopy):
                    for i in range(1, len(embed)):
                        embedArray += [float(embed[i])]
                    self.largeParagramEmbeding[curemb] = embedArray
                    wcopy.remove(curemb)
                if (not wcopy):
                    break
        print "Total of " + str(len(words) - len(wcopy)) + " words found"




class fancySentences:
    def __init__(self, s1, s2, a):
        self.sentence1 = s1
        self.sentence2 = s2
        self.alignment = a


class fancyPredict:
    def __init__(self):
        self.alignedSentencePairs = []
        self.currentSentencePair = None

    def addAlignedSents(self, sent1, sent2, align):
        self.alignedSentencePairs += [fancySentences(sent1, sent2, align)]

    def alignReader(self, filename):
        with open(filename, "r") as f:
            alignedData = json.load(f)

        for data in alignedData:
            sent1 = data['source']
            sent2 = data['target']
            alignment = []
            alignString = data['sureAlign'].split(' ')
            for astr in alignString:
                tp = astr.split('-')
                if (len(tp) < 2):
                    continue
                alignment += [(int(tp[0]),int(tp[1]))]
            self.addAlignedSents(sent1, sent2, alignment)
        if (len(self.alignedSentencePairs) > 0):
            self.changeCurrentSentence(self.alignedSentencePairs[0])

    def getWordsFromSentence(self, sent):
        return sent.split(' ')

    def changeCurrentSentence(self, sentPair):
        self.currentSentencePair = sentPair

    def getCurrentSentencePair(self):
        return self.currentSentencePair

    def getVocab(self):
        vocab = set()
        for spa in self.alignedSentencePairs:
            vocab.update(self.getWordsFromSentence(spa.sentence1))
            vocab.update(self.getWordsFromSentence(spa.sentence2))
        return vocab


    def getAlignedWords(self, alignedSentPair):
        alignedWords = []
        sent1 = self.getWordsFromSentence(alignedSentPair.sentence1)
        sent2 = self.getWordsFromSentence(alignedSentPair.sentence2)
        for aligncoords in alignedSentPair.alignment:
            align1, align2 = aligncoords
            alignedWords += [(sent1[align1], sent2[align2])]
        return alignedWords

class trainerAndParser:
    def __init__(self, input_file, output_file):
        self.inputFilename = input_file
        self.outputFilename = output_file
        self.paragramLoaded = False

        self.alignFilename = self.inputFilename[:-3] + "json"
        self.AlignedSentences = fancyPredict()
        self.AlignedSentences.alignReader(self.alignFilename)

        self.paragram = ParagramLoader(self.AlignedSentences.getVocab())

        self.predictedScores = []

        self.weights = {}
        self.accWeights = {}
        self.accCounter = {}

        self.AlignedClasses = ["1", "2", "3", "4", "5"]
        self.goldScores = []

        #print self.AlignedSentences.getAlignedWords(self.AlignedSentences.currentSentencePair)


    def loadParagram(self):
        self.paragram.loadParagramEmbeddingInMemory()
        self.paragram.loadBacktrackParagramEmbedding()
        self.paragramLoaded = True

    def getCosineSimilarityValues(self):
        if (not self.paragramLoaded):
            return None
        cosSim = []

        for asent in self.AlignedSentences.alignedSentencePairs:
            cosSim += [self.paragram.calcCosineSimilarity(self.AlignedSentences.getWordsFromSentence(asent.sentence1),
                                                          self.AlignedSentences.getWordsFromSentence(asent.sentence2))]
        return cosSim

    def getAlignmentSimilarity(self):
        # However you think similiarity from Alignement can be calculated
        alignSim = []
        for asent in self.AlignedSentences.alignedSentencePairs:
            nc1 = len(self.AlignedSentences.getWordsFromSentence(asent.sentence1))
            nc2 = len(self.AlignedSentences.getWordsFromSentence(asent.sentence2))
            ncaa = set()
            ncab = set()
            for align in asent.alignment:
                a, b = align
                ncaa.add(a)
                ncab.add(b)
            nca1 = len(ncaa)
            nca2 = len(ncab)
            SultanScore = (float(nca1 + nca2) / float(nc1 + nc2))
            alignSim += [SultanScore]
        return alignSim

    def getAveragedSim(self, weight):
        asim = self.getAlignmentSimilarity()
        if (not self.paragramLoaded):
            return asim
        csim = self.getCosineSimilarityValues()
        tsim = []
        for i in range(0, len(asim)):
            tsim += [((asim[i] * (1.0 - weight)) + (csim[i] * weight))]
        return tsim

    def savePrediction(self):
        f = open(self.outputFilename, "w")
        for score in self.predictedScores:
            f.write("{:.3f}".format(score) + '\n')
        f.close()
        print "Done saving output file"

    def setPredictedScore(self, pscore):
        self.predictedScores = pscore

    def updateOutputFile(self, filename):
        self.outputFilename= filename

    def updateWeights(self, align, currectClass, predictedClass):
        a1, a2 = align
        if (a1, a2, currectClass) not in self.weights:
            self.weights[(a1, a2, currectClass)] = 1.0
            self.weights[(a2, a1, currectClass)] = 1.0
        else:
            self.weights[(a1, a2, currectClass)] += 1.0
            self.weights[(a2, a1, currectClass)] += 1.0

        if (a1, a2, predictedClass) not in self.weights:
            self.weights[(a1, a2, predictedClass)] = -1.0
            self.weights[(a2, a1, predictedClass)] = -1.0
        else:
            self.weights[(a1, a2, predictedClass)] -= 1.0
            self.weights[(a2, a1, predictedClass)] -= 1.0

        if (a1, a2, currectClass) not in self.accWeights:
            self.accWeights[(a1, a2, currectClass)] = self.weights[(a1, a2, currectClass)]
            self.accWeights[(a2, a1, currectClass)] = self.weights[(a2, a1, currectClass)]
            self.accCounter[(a1, a2, currectClass)] = 1.0
            self.accCounter[(a2, a1, currectClass)] = 1.0
        else:
            self.accWeights[(a1, a2, currectClass)] += self.weights[(a1, a2, currectClass)]
            self.accWeights[(a2, a1, currectClass)] += self.weights[(a2, a1, currectClass)]
            self.accCounter[(a1, a2, currectClass)] += 1.0
            self.accCounter[(a2, a1, currectClass)] += 1.0

        if (a1, a2, predictedClass) not in self.accWeights:
            self.accWeights[(a1, a2, predictedClass)] = self.weights[(a1, a2, predictedClass)]
            self.accWeights[(a2, a1, predictedClass)] = self.weights[(a2, a1, predictedClass)]
            self.accCounter[(a1, a2, predictedClass)] = 1.0
            self.accCounter[(a2, a1, predictedClass)] = 1.0
        else:
            self.accWeights[(a1, a2, predictedClass)] += self.weights[(a1, a2, predictedClass)]
            self.accWeights[(a2, a1, predictedClass)] += self.weights[(a2, a1, predictedClass)]
            self.accCounter[(a1, a2, predictedClass)] += 1.0
            self.accCounter[(a2, a1, predictedClass)] += 1.0

    def updatePerceptronWeights(self, currectClass, predictedClass, sentence):
        awords1 = self.AlignedSentences.getWordsFromSentence(sentence.sentence1)
        awords2 = self.AlignedSentences.getWordsFromSentence(sentence.sentence2)
        for align in sentence.alignment:
            aw = (awords1[align[0]], awords2[align[1]])
            self.updateWeights(aw, currectClass, predictedClass)

    def trainPerceptron(self, trainFilename):
        trainAlignFilename = trainFilename[:-3] + "json"
        trainAlignedSentences = fancyPredict()
        trainAlignedSentences.alignReader(trainAlignFilename)

        for i in range(0, len(trainAlignedSentences.alignedSentencePairs)):
            aSentPairs = trainAlignedSentences.alignedSentencePairs[i]
            awords = trainAlignedSentences.getAlignedWords(aSentPairs)
            numNotAligned = self.determineNotAlignedWords(aSentPairs)
            predictedScore = self.predictPerceptronScore(awords, numNotAligned)
            if (predictedScore != self.goldScores[i]):
                self.updatePerceptronWeights(self.goldScores[i], predictedScore, aSentPairs)

        for w in self.weights:
            self.weights[w] = self.accWeights[w] / self.accCounter[w]

    def getWeight(self, alignmentPair, alignClass):
        a1, a2 = alignmentPair
        if (a1, a2, alignClass) in self.weights:
            return self.weights[(a1, a2, alignClass)]
        elif (alignClass == self.AlignedClasses[2]):
            return 1.0
        else:
            return 0.0


    def predictPerceptronScore(self, alignementPairs, notAligned):
        predictedClass = self.AlignedClasses[0]
        maxWeight = -100000.0
        for c in self.AlignedClasses:
            if (c == self.AlignedClasses[0]):
                weightSum = float(notAligned)
            else:
                weightSum = 0.0
            for align in alignementPairs:
                weightSum += self.getWeight(align, c)
            if (weightSum > maxWeight):
                maxWeight = weightSum
                predictedClass = c

        return predictedClass

    def parsePerceptron(self):
        predicition = []
        for sent in self.AlignedSentences.alignedSentencePairs:
            awords = self.AlignedSentences.getAlignedWords(sent)
            numNotAligned = self.determineNotAlignedWords(sent)
            predicition += [float(self.predictPerceptronScore(awords, numNotAligned))]
        return predicition

    def readGoldScore(self, filename):
        self.goldScores = []
        scoreList = []
        with open(filename, "r") as f:
            for line in f:
                scoreList += [float(line)]

        for score in scoreList:
            for c in self.AlignedClasses:
                if (score <= float(c) ):
                    self.goldScores += c
                    break

    def determineNotAlignedWords(self,aSentPair):
        awords1 = self.AlignedSentences.getWordsFromSentence(aSentPair.sentence1)
        awords2 = self.AlignedSentences.getWordsFromSentence(aSentPair.sentence2)
        aligned1 = set()
        aligned2 = set()
        for a1, a2 in aSentPair.alignment:
            aligned1.add(a1)
            aligned2.add(a2)

        numNotAligned = 0
        for w in awords1:
            if w not in aligned1:
                numNotAligned += 1

        for w in awords2:
            if w not in aligned2:
                numNotAligned += 1

        return numNotAligned



if __name__ == "__main__":

    if len(sys.argv) < 3:
        sys.exit(
            "Too few input arguments! The script requires 2 file names as arguments: input_data, output_score")

    (input_file, output_file) = sys.argv[1:]

    #input_file = "train\STS2012-en-train\STS.input.MSRpar.txt"
    # calcScoresFancy("train\STS2012-en-train\STS.input.MSRpar.txt", "res.txt")
    app = trainerAndParser(input_file, output_file)
    app.loadParagram()

    for i in range(0, 5):
        app.readGoldScore("train\STS2012-en-train\STS.gs.MSRpar.txt")
        app.trainPerceptron("train\STS2012-en-train\STS.input.MSRpar.txt")
        app.readGoldScore("train\STS2012-en-train\STS.gs.MSRvid.txt")
        app.trainPerceptron("train\STS2012-en-train\STS.input.MSRvid.txt")
        app.readGoldScore("train\STS2012-en-train\STS.gs.SMTeuroparl.txt")
        app.trainPerceptron("train\STS2012-en-train\STS.input.SMTeuroparl.txt")
        app.readGoldScore("train\STS2012-en-test\STS.gs.MSRpar.txt")
        app.trainPerceptron("train\STS2012-en-test\STS.input.MSRpar.txt")
        app.readGoldScore("train\STS2012-en-test\STS.gs.MSRvid.txt")
        app.trainPerceptron("train\STS2012-en-test\STS.input.MSRvid.txt")
        app.readGoldScore("train\STS2012-en-test\STS.gs.SMTeuroparl.txt")
        app.trainPerceptron("train\STS2012-en-test\STS.input.SMTeuroparl.txt")
        app.readGoldScore("train\STS2012-en-test\STS.gs.surprise.OnWN.txt")
        app.trainPerceptron("train\STS2012-en-test\STS.input.surprise.OnWN.txt")
        app.readGoldScore("train\STS2012-en-test\STS.gs.surprise.SMTnews.txt")
        app.trainPerceptron("train\STS2012-en-test\STS.input.surprise.SMTnews.txt")
        #app.readGoldScore("train\STS2013-en-test\STS.gs.FNWN.txt")
        #app.trainPerceptron("train\STS2013-en-test\STS.input.FNWN.txt")
        app.readGoldScore("train\STS2013-en-test\STS.gs.headlines.txt")
        app.trainPerceptron("train\STS2013-en-test\STS.input.headlines.txt")
        app.readGoldScore("train\STS2013-en-test\STS.gs.OnWN.txt")
        app.trainPerceptron("train\STS2013-en-test\STS.input.OnWN.txt")
        app.readGoldScore("train\STS2014-en-test\STS.gs.deft-forum.txt")
        app.trainPerceptron("train\STS2014-en-test\STS.input.deft-forum.txt")
        app.readGoldScore("train\STS2014-en-test\STS.gs.deft-news.txt")
        app.trainPerceptron("train\STS2014-en-test\STS.input.deft-news.txt")
        app.readGoldScore("train\STS2014-en-test\STS.gs.headlines.txt")
        app.trainPerceptron("train\STS2014-en-test\STS.input.headlines.txt")
        app.readGoldScore("train\STS2014-en-test\STS.gs.images.txt")
        app.trainPerceptron("train\STS2014-en-test\STS.input.images.txt")
        app.readGoldScore("train\STS2014-en-test\STS.gs.OnWN.txt")
        app.trainPerceptron("train\STS2014-en-test\STS.input.OnWN.txt")
        app.readGoldScore("train\STS2014-en-test\STS.gs.tweet-news.txt")
        app.trainPerceptron("train\STS2014-en-test\STS.input.tweet-news.txt")
        app.readGoldScore("train\STS2015-en-test\STS.gs.answers-forums.txt")
        app.trainPerceptron("train\STS2015-en-test\STS.input.answers-forums.txt")
        app.readGoldScore("train\STS2015-en-test\STS.gs.answers-students.txt")
        app.trainPerceptron("train\STS2015-en-test\STS.input.answers-students.txt")
        app.readGoldScore("train\STS2015-en-test\STS.gs.belief.txt")
        app.trainPerceptron("train\STS2015-en-test\STS.input.belief.txt")
        app.readGoldScore("train\STS2015-en-test\STS.gs.headlines.txt")
        app.trainPerceptron("train\STS2015-en-test\STS.input.headlines.txt")
        app.readGoldScore("train\STS2015-en-test\STS.gs.images.txt")
        app.trainPerceptron("train\STS2015-en-test\STS.input.images.txt")



    avgScore = []
    scorePerc = app.parsePerceptron()
    scoreSim = app.getAveragedSim(1.0)
    #scoreSim = app.parsePerceptron()

    #avgScore = []
    for i in range(0, len(scorePerc)):
        avgScore += [((scorePerc[i] * (0.2) * (0.2)) + (scoreSim[i] * 0.8))]
    app.setPredictedScore(avgScore)
    #app.setPredictedScore(scoreSim)
    app.savePrediction()

    #for j in range(0, 11):
    #    avgScore = []
    #    outputfile = "res" + str(j) + ".txt"
    #    app.updateOutputFile(outputfile)
    #    fi = float(j) / 10.0
    #    for i in range(0, len(scorePerc)):
    #        avgScore += [((scorePerc[i] * (0.2) * (1 - fi)) + (scoreSim[i] * fi))]
    #    app.setPredictedScore(avgScore)
    #    app.savePrediction()

    #for i in range(0, 11, 1):
    #    fi = float(i) / 10.0
    #    outputfile = "res" + str(i) + ".txt"
    #    app.updateOutputFile(outputfile)
    #    app.setPredictedScore(app.getAveragedSim(fi))
    #    app.savePrediction()


    #print app.getAeragedSim(0.5)
    #print app.getCosineSimilarityValues()
