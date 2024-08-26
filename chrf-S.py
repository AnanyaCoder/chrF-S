import sys
import math
import unicodedata
import argparse
from collections import defaultdict
import time
import string
from sentence_transformers import SentenceTransformer, util

from sklearn import preprocessing
from scipy.stats import pearsonr
import numpy
from numpy import dot
from numpy.linalg import norm
model = SentenceTransformer('LaBSE')

def normalize_min_max(score_list,minvalue,maxvalue):
    scalar = preprocessing.MinMaxScaler(feature_range=(minvalue, maxvalue))
    d = scalar.fit_transform(score_list.values.reshape(-1, 1))
    finalres = [item[0] for item in d]

    return finalres

def pearson_correlation(x,y):
    norm_X = normalize_min_max(x,0,1)
    norm_Y = normalize_min_max(y,0,1)
    eval_pearson_cosine, _ = pearsonr(norm_X,norm_Y)
    #print(norm_X)
    #print(norm_Y)
    return eval_pearson_cosine

def getSentenceSimScore(model,sen_list):
    cos_sim = []
    embeddings = []
    #Compute sentence embeddings using sentence embedding model like LaBSE, XLM-Roberta-base etc.. 
    embeddings = model.encode(sen_list)
    #Compute cosine-similarities
    for k in range(1,len(sen_list)):
        cos_sim = dot(embeddings[0], embeddings[k])/(norm(embeddings[0])*norm(embeddings[k]))  
    return cos_sim

def getChrf_S(sentF,sent_sim_score):
    denom = sentF + sent_sim_score
    if denom > 0:
        fscore = 2* sentF * sent_sim_score / denom
    else:
        fscore = 1e-16
        
    return fscore
def separate_characters(line):
    return list(line.strip().replace(" ", ""))

def separate_punctuation(line):
    words = line.strip().split()
    tokenized = []
    for w in words:
        if len(w) == 1:
            tokenized.append(w)
        else:
            lastChar = w[-1] 
            firstChar = w[0]
            if lastChar in string.punctuation:
                tokenized += [w[:-1], lastChar]
            elif firstChar in string.punctuation:
                tokenized += [firstChar, w[1:]]
            else:
                tokenized.append(w)
    
    return tokenized
    
def ngram_counts(wordList, order):
    counts = defaultdict(lambda: defaultdict(float))
    nWords = len(wordList)
    for i in range(nWords):
        for j in range(1, order+1):
            if i+j <= nWords:
                ngram = tuple(wordList[i:i+j])
                counts[j-1][ngram]+=1
   
    return counts

def ngram_matches(ref_ngrams, hyp_ngrams):
    matchingNgramCount = defaultdict(float)
    totalRefNgramCount = defaultdict(float)
    totalHypNgramCount = defaultdict(float)
 
    for order in ref_ngrams:
        for ngram in hyp_ngrams[order]:
            totalHypNgramCount[order] += hyp_ngrams[order][ngram]
        for ngram in ref_ngrams[order]:
            totalRefNgramCount[order] += ref_ngrams[order][ngram]
            if ngram in hyp_ngrams[order]:
                matchingNgramCount[order] += min(ref_ngrams[order][ngram], hyp_ngrams[order][ngram])


    return matchingNgramCount, totalRefNgramCount, totalHypNgramCount


def ngram_precrecf(matching, reflen, hyplen, beta):
    ngramPrec = defaultdict(float)
    ngramRec = defaultdict(float)
    ngramF = defaultdict(float)
    
    factor = beta**2
    
    for order in matching:
        if hyplen[order] > 0:
            ngramPrec[order] = matching[order]/hyplen[order]
        else:
            ngramPrec[order] = 1e-16
        if reflen[order] > 0:
            ngramRec[order] = matching[order]/reflen[order]
        else:
            ngramRec[order] = 1e-16
        denom = factor*ngramPrec[order] + ngramRec[order]
        if denom > 0:
            ngramF[order] = (1+factor)*ngramPrec[order]*ngramRec[order] / denom
        else:
            ngramF[order] = 1e-16
            
    return ngramF, ngramRec, ngramPrec

#def computeChrF(fpRef, fpHyp, nworder, ncorder, beta, sentence_level_scores = None):

#***********chrf++S start*****************************************
def computeChrF(fpRef, fpHyp, nworder, ncorder, beta, model, sentence_level_scores = None):
#***********chrf++S end*****************************************
    norder = float(nworder + ncorder)

    # initialisation of document level scores
    totalMatchingCount = defaultdict(float)
    totalRefCount = defaultdict(float)
    totalHypCount = defaultdict(float)
    totalChrMatchingCount = defaultdict(float)
    totalChrRefCount = defaultdict(float)
    totalChrHypCount = defaultdict(float)
    averageTotalF = 0.0
    #***********chrf++S start*****************************************
    sent_chrf_score = []
    sent_chrfS_HM_score = []
    sent_chrfS_AM_score = []
    #***********chrf++S end*****************************************
    nsent = 0
    for hline, rline in zip(fpHyp, fpRef):
        nsent += 1
        
        # preparation for multiple references
        maxF = 0.0
        bestMatchingCount = 0
        bestWordMatchingCount = None
        bestCharMatchingCount = None
        
        hypNgramCounts = ngram_counts(separate_punctuation(hline), nworder)
        hypChrNgramCounts = ngram_counts(separate_characters(hline), ncorder)

        # going through multiple references

        refs = rline.split("*#")

        for ref in refs:
            refNgramCounts = ngram_counts(separate_punctuation(ref), nworder)
            refChrNgramCounts = ngram_counts(separate_characters(ref), ncorder)

            # number of overlapping n-grams, total number of ref n-grams, total number of hyp n-grams
            matchingNgramCounts, totalRefNgramCount, totalHypNgramCount = ngram_matches(refNgramCounts, hypNgramCounts)
            matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount = ngram_matches(refChrNgramCounts, hypChrNgramCounts)
                    
            # n-gram f-scores, recalls and precisions
            ngramF, ngramRec, ngramPrec = ngram_precrecf(matchingNgramCounts, totalRefNgramCount, totalHypNgramCount, beta)
            chrNgramF, chrNgramRec, chrNgramPrec = ngram_precrecf(matchingChrNgramCounts, totalChrRefNgramCount, totalChrHypNgramCount, beta)

            sentRec  = (sum(chrNgramRec.values())  + sum(ngramRec.values()))  / norder
            sentPrec = (sum(chrNgramPrec.values()) + sum(ngramPrec.values())) / norder
            sentF    = (sum(chrNgramF.values())    + sum(ngramF.values()))    / norder

            if sentF > maxF:
                maxF = sentF
                bestMatchingCount = matchingNgramCounts
                bestRefCount = totalRefNgramCount
                bestHypCount = totalHypNgramCount
                bestChrMatchingCount = matchingChrNgramCounts
                bestChrRefCount = totalChrRefNgramCount
                bestChrHypCount = totalChrHypNgramCount
        # all the references are done
            #***********chrf++S start *****************************************
            if(model):
                sent_sim_score = getSentenceSimScore(model,[hline,ref])
                ##print('hi',sent_sim_score)
                #sent_chrfS_HM_score.append(getChrf_S(maxF,sent_sim_score))
                sent_chrfS_AM_score.append((maxF + sent_sim_score)/2)
            else:
                sent_chrfS_HM_score = []
                sent_chrfS_AM_score = []
                
            #***********chrf++S end *****************************************

        # write sentence level scores
        if sentence_level_scores:
            #sentence_level_scores.write("%i::c%i+w%i-F%i\t%.4f\n"  % (nsent, ncorder, nworder, beta, 100*maxF))
            #***********chrf++S start*****************************************
            sent_chrf_score.append(100*maxF)
            #***********chrf++S end*****************************************
        '''
        # collect document level ngram counts
        for order in range(nworder):
            totalMatchingCount[order] += bestMatchingCount[order]
            totalRefCount[order] += bestRefCount[order]
            totalHypCount[order] += bestHypCount[order]
        for order in range(ncorder):
            totalChrMatchingCount[order] += bestChrMatchingCount[order]
            totalChrRefCount[order] += bestChrRefCount[order]
            totalChrHypCount[order] += bestChrHypCount[order]

        averageTotalF += maxF

    # all sentences are done
     
    # total precision, recall and F (aritmetic mean of all ngrams)
    totalNgramF, totalNgramRec, totalNgramPrec = ngram_precrecf(totalMatchingCount, totalRefCount, totalHypCount, beta)
    totalChrNgramF, totalChrNgramRec, totalChrNgramPrec = ngram_precrecf(totalChrMatchingCount, totalChrRefCount, totalChrHypCount, beta)

    totalF    = (sum(totalChrNgramF.values())    + sum(totalNgramF.values()))    / norder
    averageTotalF = averageTotalF / nsent
    totalRec  = (sum(totalChrNgramRec.values())  + sum(totalNgramRec.values()))  / norder
    totalPrec = (sum(totalChrNgramPrec.values()) + sum(totalNgramPrec.values())) / norder
    '''
    return sent_chrfS_AM_score,sent_chrf_score

'''
def chrf(htxt,rtxt,chrf_type,model):
    sentence_level_scores = None
    #model = SentenceTransformer('LaBSE')
    sent_chrf_S_AM,sent_chrf_score  = computeChrF(rtxt, htxt, 2, 6, 2.0, model, sentence_level_scores)
    
    return sent_chrf_S_AM

'''   
    
def main():
    sys.stdout.write("start_time:\t%i\n" % (time.time()))


    argParser = argparse.ArgumentParser()
    argParser.add_argument("-R", "--reference", help="reference translation", required=True)
    argParser.add_argument("-H", "--hypothesis", help="hypothesis translation", required=True)
    argParser.add_argument("-nc", "--ncorder", help="character n-gram order (default=6)", type=int, default=6)
    argParser.add_argument("-nw", "--nworder", help="word n-gram order (default=2)", type=int, default=2)
    argParser.add_argument("-b", "--beta", help="beta parameter (default=2)", type=float, default=2.0)
    argParser.add_argument("-s", "--sent", help="show sentence level scores", action="store_true")

    args = argParser.parse_args()

    rtxt = open(args.reference, 'r')
    htxt = open(args.hypothesis, 'r')

    sentence_level_scores = None
    if args.sent:
        sentence_level_scores = sys.stdout # Or stderr?

    #totalF, averageTotalF, totalPrec, totalRec = computeChrF(rtxt, htxt, args.nworder, args.ncorder, args.beta, sentence_level_scores)

    #sys.stdout.write("c%i+w%i-F%i\t%.4f\n"  % (args.ncorder, args.nworder, args.beta, 100*totalF))
    #sys.stdout.write("c%i+w%i-avgF%i\t%.4f\n"  % (args.ncorder, args.nworder, args.beta, 100*averageTotalF))
    #sys.stdout.write("c%i+w%i-Prec\t%.4f\n" % (args.ncorder, args.nworder, 100*totalPrec))
    #sys.stdout.write("c%i+w%i-Rec\t%.4f\n"  % (args.ncorder, args.nworder, 100*totalRec))

    chrfS,chrf  = computeChrF(rtxt, htxt, 2, 6, 2.0, model, sentence_level_scores)

    #with open('chrf-s_Scores.txt','w') as f:
    #for line in chrfS:
        #print(line)
    with open('chrf-S_Scores.txt', 'w') as f:
        for line in chrfS:
    #for line in chrfS:
            f.write(f"{line}\n")
    
    sys.stdout.write("end_time:\t%i\n" % (time.time()))

    htxt.close()
    rtxt.close()


if __name__ == "__main__":
    main()
