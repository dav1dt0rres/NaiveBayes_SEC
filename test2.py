from __future__ import print_function, unicode_literals, division

import math
import random
import warnings
import array
import urllib
import string
import re
import requests
import itertools
import nltk
from nltk.classify import NaiveBayesClassifier
import decimal
from decimal import *
from operator import itemgetter
from collections import defaultdict, Counter
from functools import reduce
from nltk.stem.porter import PorterStemmer 
porter_stemmer=PorterStemmer()
import numpy as np


def Parser(line):
	tokenized_text=nltk.word_tokenize(line)
	return tokenized_text
	
def Basic_Collection(lines):
	IRS=0
	SIC=0
	Fiscal=0
	Fiscal_1=0
	for line in lines:
		Array=Parser(line)
		if IRS==0 or SIC==0 or Fiscal==0 or Fiscal_1==0:
			if "IRS" in Array:
				for x in Array:
					if x.isdigit():
						IRS=int(x)
				
				#print("IRS",IRS)
			if "CLASSIFICATION" in Array :
				for x in Array:
					if x.isdigit():
						SIC=int(x)
						
					
			if "PERIOD" in Array:
				length=len(Array)
				Fiscal=Array[length-1]
				#print("Fiscal",Fiscal)
			if "CONFORMED" in Array and "PERIOD" in Array:
				for x in Array:
					if x.isdigit():
						Fiscal_1=int(x)
	return IRS,SIC,Fiscal,Fiscal_1
	
def Paragraph(lines):
	Array=[]
	for line in lines:
		line=line.lower()
		Array.append(line)
	
	Array_Final=[]
	for i in range(0,len(Array)-5):
		Array_Final.append(Array[i]+Array[i+1]+Array[i+2]+Array[i+3])
	#returns an array where each unit is a four sentence section 
	
		
		
	return Array_Final	
		
def Dictionary(Array):
#Creates a dictionary from a proxy document, "testing_test", this should be part of the training set. 
	
	Final = [word for word in Array if len(word) > 3]
	fdist = nltk.FreqDist(Final)
	fbigram=nltk.FreqDist(nltk.bigrams(Final))
	
	return fdist,fbigram


def Training(lines,monogram,bigram):
#makes the features vector based off of the features of every 4 lines. This also has the mapping output, so function used for training/testing
	features_set=[]
	Array=[]
	
	for line in lines:
		line=line.lower()
		Array.append(line)
	Array_Final=[]
	print("number of lines in entire document ",len(Array))
	for i in range(0,(len(Array)//5)-1):
		Count=[]
		Count_B=0
		#print("iteration",i)
		#print("Array itemized",Array[5*i]+Array[5*i+1]+Array[5*i+2]+Array[5*i+3])
		Array_Final.append(Array[5*i]+Array[5*i+1]+Array[5*i+2]+Array[5*i+3])
		Classification=Array[5*i+4]
		print("set of four ",Array_Final[i], "classification",Classification)
		tokenized_text=nltk.word_tokenize(Array_Final[i])
		#mongram features
		for j in range(0,len(monogram)):
			#print("searching for monogram",word,tokenized_text.count(word))
			print("monogram",monogram[j])
			Count.append(tokenized_text.count(monogram[j]))
			
			print("Count",Count[j])
		#bigram features
		for word in bigram:
			fbigram=nltk.FreqDist(nltk.bigrams(tokenized_text))
			
			for big,freq in fbigram.most_common():
			#	print("searching for bigram",big,"current bigram list",word)
				if word==big:
					print("found bigram",word,big)
					Count_B+=1
		print("Count_B",Count_B)
		#length feature	
		#print("length of four lines",len(tokenized_text))
		j=0
		f=0
		h=0
		while j<len(Count):
			if Count[j]>0:
				f+=1
			if Count[j]>=2:
				h+=1
			j+=1
		if f>1:
			print("more than one monogram found")
			feature_1=1
		else:
			
			feature_1=0
		if h>=1:
			print("found heavy monogram match")
			feature_2=1
		else:
			feature_2=0
		features_set=[({'Count':feature_1,'Count2':feature_2,'Count_B':Count_B},Classification) ]+features_set
	return features_set	
def Decision_Function(classifier,Array,monogram,bigram):
	Relevant_Array=[]
	
	for i in range(0,len(Array)):
		
		Count=[]
		Count_B=0
		features_set=[]
		tokenized_text=nltk.word_tokenize(Array_Final[i])
		#print("set of four ",Array_Final[i])
		for j in range(0,len(monogram)):
			#print("searching for monogram",word,tokenized_text.count(word))
			#print("monogram",monogram[j])
			Count.append(tokenized_text.count(monogram[j]))
			#print("Count",Count[j])
		for word in bigram:
			fbigram=nltk.FreqDist(nltk.bigrams(tokenized_text))
			for big,freq in fbigram.most_common():
				if word== big:
					#print("found bigram",word,big)
					Count_B+=1
		#print ("count_B",Count_B)			
		j=0
		f=0
		h=0
		while j<len(Count):
			if Count[j]>0:
				f+=1
			if Count[j]>=2:
				h+=1
			j+=1
		if f>1:
			feature_1=1
		else:
			feature_1=0
		if h>=1:
			feature_2=1
		else:
			feature_2=0
		
		features_set={'Count':feature_1,'Count2':feature_2,'Count_B':Count_B} 
		#print ("Final classification",classifier.classify(features_set)	)
		if int(classifier.classify(features_set))==1:
			Relevant_Array.append(Array[i])
		
	return Relevant_Array
def Distance_Function(words,Root,Set):
	#print ("set",Set)
	Net=np.array=[]
	tokenized_text=nltk.word_tokenize(Set)
	
	for i in range(0,len(words)):	
		#print("looking for ",words[i])
		#print("count",tokenized_text.count(words[i]))
		for u in range(0,len(tokenized_text)):
			if porter_stemmer.stem(tokenized_text[u])==porter_stemmer.stem(words[i]) :
				Net.append(u)
				#print("deleted",tokenized_text[u])
				del tokenized_text[u]
				
				break
				#print("apeneded",tokenized_text.index(words[i]))
	
	for i in range(0,len(Net)-1):	
		#print("comparing indexs",Net[i],Net[i+1])

		if abs(Net[i]-Net[i+1]) > 4:
			#print("not close enough")
			#raw=raw_input()
			return False
	return True	
	
def Output_Function(str_4,str_5,sitename,Company,CIK,SIC,IRS,Year,Fiscal,Array):
	#outputs an array of integers that represent the classifications for a specific paragraph		
	OpenFile_4=open(str_4,'a')
	OpenFile_5=open(str_5,'a')
	Dictionary_1=[['coverage','ratio'], ['interest','coverage'],['debt', 'service', 'coverage']]
	Dictionary_2=[['fixed','charge','coverage']]
	Dictionary_3=[['ratio','of','debt','ebitda'],['ratio','of','debt','ebitdar'],['ratio','of','debt','ebitdax'],['debt',' to',' ebitda'],['debt','to','ebitdar'],['debt','to','ebitdax'],['ratio', 'debt',' to',' net',' income'],["leverage","ratio"]]
	
	
	Dictionary_4=[['minumum','ebitda'],['minimum','ebitdar'],['minumum','ebitdax'],['minimum net income']]
	#Classificaiton:6	
	Dictionary_6=[['book','equity','ratio'],['net','worth','to','asset'],['debt','to', 'equity'],['net' ,'equity' ,'ratio'],['equity','to','asset'],['book', 'leverage'], ['debt','to',' asset'],['debt','to','equity'],['debt','to','net','worth'],['net','worth','to','asset'],['net','worth','ratio'],['liabilities', 'to', 'net', 'worth'],['liabilities','to','equity']]
	#Classification:5
	Dictionary_5=[['minimum', 'net', 'worth'],['minumum','tangible','net','worth'],['minimum','net','equity']]	
	
	Dictionary_7=[['maintain', 'working','capital'],['maintain','inventory'],['maintain','receivable']]
	Dictionary_12=[['mortgage','back']]
	Dictionary_8=[['mortgage','notes']]
	Dictionary_9=[['all','assets','liens'],['secured','by','all','assets'],['all', 'assets','collateralized'],['all','assets','security']]
	Dictionary_10=[['convertible','bond'],['convertible','debt'],[ 'convertible' ,'debenture'],['convertible', 'note'],[ 'convertible',' subordinated'],['convertible', 'secured'],[ 'convertible' ,'senior']]
	Dictionary_11=[['financial', 'covenant'], ['loan', 'covenant'],[ 'debt', 'covenant'],[ 'restrictive', 'covenant'],['covenant','maintain'],['covenant','ratio'],['covenant','minumum'],['require','covenant'],['contain','covenant'],['maintain','covenant'],['minimum','covenant'],['covenant','ebitda'],['net','worth','covenant'],['coverage','covenant']]
	
	
	
	
	Dictionary=	[Dictionary_1,Dictionary_2,Dictionary_3,Dictionary_4,Dictionary_5,Dictionary_6,Dictionary_7,Dictionary_8,Dictionary_9,Dictionary_10,Dictionary_11,Dictionary_12]
	Output_Array=[]
	
	
	
	#Array has all the relevant paragraphs of the document
	for i in range(0,len(Array)):
		#print("This is original text",Array[i])
	
		tokenized_text=nltk.word_tokenize(Array[i])
		Root=[]
		for token in tokenized_text: 
			Root.append(porter_stemmer.stem(token))
		Root=list(set(Root))
		#print ("Root",Root)
		
		for j in range(0,len(Dictionary)):
			
			for k in Dictionary[j]:
			
				Count=0
				for word in k:
					word=porter_stemmer.stem(word)
					
					Count=Root.count(word)+Count
					#print("Matches for word",word,Count)
					
				if Count==len(k) and Distance_Function(k,Root,Array[i]):
					Output_Array.append(j+1)
					
					#print("found a match for Classification:",j+1)
					#raw=raw_input()

	Output_Array=set(Output_Array)
	print("OUtput Array",Output_Array)
	if not Output_Array:
		print("List is empty",Company)
		
	else:	
		#capone = np.array(Output_Array)
	
		
		np.savetxt(OpenFile_4, np.column_stack(Output_Array), delimiter=',')
		OpenFile_5.write('%s%s%s%s%s%s%s%s%s%s%s%s%s%s'%('\n',Company,",",SIC,",",IRS,",",CIK,",",Year,",",Fiscal,",","https://www.sec.gov/Archives/"+sitename))
		OpenFile_5.close()
		OpenFile_4.close()

	

#str= raw_input("Hello Yueran, please type the dictionary file ")
str='C:\Python27\Dictionary_Doc.txt'
#str_1=raw_input("Please type the sec.gove edgar file ")
str_1='C:\Python27\Company_SEC.txt'
#str_2=raw_input("PLease type the training file")
str_2='C:\Python27\Training_Doc.txt'
#str_3=raw_input("Please type the testing file")	
str_3='C:\Python27\Testing_Doc.txt'
#str_4=raw_input("Please type the Output File")
str_4='C:\Python27\Classification_Training.txt'
OpenFile = open(str_1,'r')
lines = OpenFile.readlines()#Firm information

OpenFile_1=open(str,'r') #Dictionary text documnet
lines_1=OpenFile_1.readlines()

OpenFile_2=open(str_2,'r')#Training document
lines_2=OpenFile_2.readlines()
OpenFile_3=open(str_3,'r')#testing document
lines_3=OpenFile_3.readlines()
OpenFile_4=open(str_4,'w')
str_5='C:\Python27\Classification_Training_1.txt'
OpenFile_5=open(str_5,'w')

Array_2=[]		
		
for line in lines_1:
	line=line.lower()
	Array=Parser(line)
	Array_2=Array_2+Array
	

monograms,bigrams=Dictionary(Array_2) # This is the dictionary creationg later to be used to get features for training data and testing data
Delete=[]
Delete_bigram=[]
#it takes the most common from the frquency list, and inserts into a normal list and also taking away certain bigrams that should be thrown out
for a,b in monograms.most_common(15):
	Delete.append(a)
for a,b in bigrams.most_common(15):
	if a==('with','respect') or a==('common','stock') or a==('with','certain') or a==('common','stock'):
		
		print("Deleting bigram cycling",a,b)
	
	else:
		Delete_bigram.append(a)
	
Delete.remove('company')	
Delete.remove('1998')
Delete.remove('september')
Delete.remove('under')
Delete.remove('with')	
Delete.remove('certain')
Delete.remove('million')
Delete.append('covenants')
Delete.append('convenant')
Delete.append('mortgage')
Delete.append('mortgage notes')
Delete.append('collateralized')
Delete.append('collateral')


for first in Delete_bigram:
	
	print("bigrams",first)
for first in Delete:
	print("monograms",first)
#The training function takees in the lines from test .txt and list of single and double most common from dictionary above
train_set=Training(lines_2,Delete,Delete_bigram)
classifier=NaiveBayesClassifier.train(train_set)

#Testing data points from .txt document
print("NOW ENTERING THE TESTING SET")
test_set=Training(lines_3,Delete,Delete_bigram)
#prints the accuracy of the model
print(nltk.classify.accuracy(classifier, test_set))	
str= raw_input("This is the accuracy rating")	

	
for line in lines:
	sitename_1=line.split(",")[4].strip().lower()
	CIK=line.split(",")[2].strip().lower()
	Form_Type=line.split(",")[0].strip().lower()
	Company_Name=line.split(",")[1].strip().lower()
	print("company name",Company_Name)
	Year=line.split(",")[3].strip().lower()
	if Form_Type=="10-k" or Form_Type=="10-k405":
		
		sitename_final="https://www.sec.gov/Archives/"+sitename_1
		print("Sitename",sitename_final)
		print("CIK",CIK,"Form",Form_Type,"Comp Name",Company_Name,"Year",Year)
		rr= requests.get(sitename_final)
		rrr=rr.content
		
		path1="C:\Python27\ex_1"+"_"+"sample"
		fff=open(path1,"w")
	
		fff.write(rrr)
		fff.close()
		
		rrr=open(path1,"r")
		IRS,SIC,Fiscal,Fiscal_1=Basic_Collection(rrr.readlines())
		rrr.close()
		
		print ("IRS,",IRS,"SIC",SIC)
		
		if SIC>=6000 and SIC<=6999: 
			print("Ignore this SIC")
		elif SIC>=9000 and SIC<=9999:
			print("Ignore this SIC")
		else:
			aaa=open(path1,"r")
			Array_Final=Paragraph(aaa.readlines())
			
			#should output an array of relevant 4 line sets. For ONE document
			Relevant_Array=Decision_Function(classifier,Array_Final,Delete,Delete_bigram)
			
			for u in range(0,len(Relevant_Array)):
				print("These are all the relevant paragrapgs from the document before it classifies..",Relevant_Array[u])
				
			Output_Function(str_4,str_5,sitename_1,Company_Name,CIK,SIC,IRS,Year,Fiscal,Relevant_Array)
		

