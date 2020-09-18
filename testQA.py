from DocumentRetrievalModel import DocumentRetrievalModel
from ProcessedQuestion import ProcessedQuestion

from DocumentRetrievalModelWM import DocumentRetrievalModelWM
from ProcessedQuestionWM import ProcessedQuestionWM

from StanfordDataset import StanfordDataset
from nltk.tokenize import word_tokenize
import csv
import math
import argparse

from gensim.models import Word2Vec

def computeAccuracy(topic, use_word_embeddings, sd = StanfordDataset()):
    
    testPara = sd.getParagraph(topic)

	# Loading Model
    modelName = f"models/{topic}.h5"
    model =  Word2Vec.load(modelName)

    if use_word_embeddings:
        drm = DocumentRetrievalModelWM(testPara, model, True,True)
    else:
        drm = DocumentRetrievalModel(testPara, True, True)
    
    result = []
    res = [[0,0],[0,0],[0,0],[0,0]]
    devData =sd.getTopic(topic)
    for index in range(0,len(devData['paragraphs'])):
        p = devData['paragraphs'][index]
        for qNo in range(0,len(p['qas'])):
            if use_word_embeddings:
                pq = ProcessedQuestionWM(p['qas'][qNo]['question'], model, True,False,True)
            else:
                pq = ProcessedQuestion(p['qas'][qNo]['question'], True,False,True)

            index = 0
            if pq.aType == 'PERSON':
                index = 0
            elif pq.aType == 'DATE':
                index = 1
            elif pq.aType == 'LOCATION':
                index = 2
            else:
                index = 3
            res[index][0] += 1
            r = drm.query(pq)
            answers = []
            for ans in p['qas'][qNo]['answers']:
                answers.append(ans['text'].lower())
            r = r.lower()
            isMatch = False
            for rt in word_tokenize(r):
                #print(rt,word_tokenize(ans) for ans in answers)
                if [rt in word_tokenize(ans) for ans in answers].count(True) > 0:
                    isMatch = True
                    res[index][1] += 1
                    break
            #if isMatch:
            #    print(pq.question,r,str(answers))
            result.append((index, qNo, pq.question, r, str(answers),isMatch))
                
    noOfResult = len(result)
    correct = [r[5] for r in result].count(True)
    if noOfResult == 0:
        accuracy = -1
    else:
        accuracy = correct/noOfResult
    #return (result,accuracy)
    #return {"Topic":topic,"No of Ques":noOfResult,"Correct Retrieval":correct,"whoAccu":res[0][1]/(res[0][0]+1),"whenAccu":res[1][1]/(res[1][0]+1),"whereAccu":res[2][1]/(res[2][0]+1),"summarizationAccu":res[3][1]/(res[3][0]+1),"OverallAccuracy":accuracy}
    return {"Topic":topic,"No of Ques":noOfResult,"Correct Retrieval":correct,"OverallAccuracy":round(accuracy*100,2)}

def runAll(use_word_embeddings):
    sd = StanfordDataset()

    toCSV = []
    total = len(sd.titles)
    index = 1
    tA = 0
    for title in sd.titles:
        print("Testing all questions for \"" + title + "\"")
        d=computeAccuracy(title, use_word_embeddings)
        if d["No of Ques"] == 0:
            continue
        tA += d['OverallAccuracy']
        print(d)
        print(str(index) + "/" + str(total) + ":",d['OverallAccuracy'],"/",tA/index)
        toCSV.append(d)
        index += 1
    print("OverallAccuracy : ",tA/total)

    keys = toCSV[0].keys()
    with open('accuracy.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)

    print("Written the accuracy measure in accuracy.csv file. Done")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='WordEmbedding.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-use_word_embeddings', action='store_true',
                       help="""Filepath to the dataset""")
    args = parser.parse_args()

    runAll(args.use_word_embeddings)