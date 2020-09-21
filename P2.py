# Author : Vaibhaw Raj
# Created on : Sun, Feb 11 2018
# Description : Entry point for Simple Question Answer Chatbot
# Program Argument :
#		datasetName = "Name of dataset text file" eg. "Beyonce.txt"
# Usage :
#		$ python3 P2.py dataset/IPod
from DocumentRetrievalModel import DocumentRetrievalModel as DRM
from ProcessedQuestion import ProcessedQuestion as PQ

from DocumentRetrievalModelWM import DocumentRetrievalModelWM as DRMWM
from ProcessedQuestionWM import ProcessedQuestionWM as PQWM

import argparse
import re
import sys

from gensim.models import Word2Vec

def main(datasetName, use_word_embeddings):
	print("Bot> Please wait, while I am loading my dependencies")
	
	# Loading Dataset
	try:
		datasetFile = open(datasetName,"r", encoding="utf-8")
	except FileNotFoundError:
		print("Bot> Oops! I am unable to locate \"" + datasetName + "\"")
		exit()

	# Retrieving paragraphs : Assumption is that each paragraph in dataset is
	# separated by new line character
	paragraphs = []
	for para in datasetFile.readlines():
		if(len(para.strip()) > 0):
			paragraphs.append(para.strip())

	# Loading Model
	modelName = datasetName.replace("dataset", "models").replace(".txt", ".h5")
	model =  Word2Vec.load(modelName)

	# Processing Paragraphs
	if use_word_embeddings:
		drm = DRMWM(paragraphs, model, True, True)
	else:
		drm = DRM(paragraphs, True, True)

	print("Bot> Hey! I am ready. Ask me factoid based questions only :P")
	print("Bot> You can say me Bye anytime you want")

	# Greet Pattern
	greetPattern = re.compile("^\ *((hi+)|((good\ )?morning|evening|afternoon)|(he((llo)|y+)))\ *$",re.IGNORECASE)

	isActive = True
	while isActive:
		userQuery = input("You> ")
		if(not len(userQuery)>0):
			print("Bot> You need to ask something")

		elif greetPattern.findall(userQuery):
			response = "Hello!"
		elif userQuery.strip().lower() == "bye":
			response = "Bye Bye!"
			isActive = False
		else:
			# Proocess Question
			if use_word_embeddings:
				pq = PQWM(userQuery, model, True,False,True)
			else:
				pq = PQ(userQuery,True,False,True)

			# Get Response From Bot
			response =drm.query(pq)
		print("Bot>",response)

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(
        description='P2.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-dataset', type=str, required=True,
                       help="""Filepath to the dataset""")
    parser.add_argument('-use_word_embeddings', action='store_true',
                       help="""Filepath to the dataset""")
    args = parser.parse_args()

    main(args.dataset, args.use_word_embeddings)