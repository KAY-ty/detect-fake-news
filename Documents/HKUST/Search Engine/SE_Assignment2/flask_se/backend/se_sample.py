'''
Author: JasonYU
Date: 2020-10-02 09:17:33
LastEditTime: 2020-10-04 10:00:43
FilePath: \SE\flask_se\backend\se_sample.py
'''

import nltk
import json
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

with open("document_word_frequency.json", "r") as f:
    document_word_frequency_dict = json.loads(f.read())

with open("word_document_frequency.json", "r") as f:
    word_document_frequecy_dict = json.loads(f.read())

with open("paper.json", "r") as f:
    paper = json.loads(f.read())

with open("Document_length.json", "r") as f:
    document_length = json.loads(f.read())

with open("processed_abstract_original_position.json", "r") as f:
    processed_abstract_dict = json.loads(f.read())

with open("unique_word.json", "r") as f:
    unique_word_dict = json.loads(f.read())

def tf(word, index):
    if word in document_word_frequency_dict[str(index)]:
        return document_word_frequency_dict[str(index)][word]
    else:
        return 0

def df(word):
    if word in word_document_frequecy_dict:
        tempdict = word_document_frequecy_dict[word]
        return len(tempdict)
    else:
        return 0

def idf(word):
    temp = df(word)
    if temp == 0:
        return 0
    else:
        return math.log(778/temp, 2)


def process_query(query):
    # Discard punctuation marks & performe tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(query)
    # Stop word remove & stemming
    stem_abstract = []
    for word in word_tokens:
        if word not in stop_words:
            stem_abstract.append(ps.stem(word))

    dict = {}
    for word in stem_abstract:
        if word in dict:
            temp = dict[word]
            dict[word] = temp+1
        else:
            dict[word] = 1
    return dict
'''
query: dict
struture:{"world": value}
'''
def get_matix(query):
    length = len(query)
    ret = np.zeros((778, length), dtype=np.float_)

    temp = []
    for word in query:
        temp.append(idf(word))

    for i in range(0,778):
        j = 0
        for word in query:
            ret[i][j] = tf(word, i)*temp[j]
            j = j+1

    return ret

'''
query: dict
struture:{"world": value}
'''
def get_vector(query):
    ret = []
    for word in query:
        ret.append(query[word])
    return ret



'''
vector: a list of query

matrix: a 2-D array related to this vector

get the similary of each document & query
'''
def CosSim(vector, matrix):
    length = len(vector)
    ret = []
    for vec in matrix:
        dq = 0
        square_d = 0
        square_p = 0
        for i in range(0, length):
            dq = dq+vec[i]*vector[i]
            square_d = square_d+vec[i]*vec[i]
            square_p = square_p+vector[i]*vector[i]
        if square_p == 0 or square_d == 0:
            ret.append(0)
        else:
            ret.append(dq/(np.sqrt(square_d)*np.sqrt(square_p)))

    return ret

'''
vector: a list of query

matrix: a 2-D array related to this vector

get the similary of each document & query
return dict
'''
def CosSim1(vector, matrix):
    length = len(vector)
    #ret = []
    ret = {}
    for index in range(0, 778):
        dq = 0
        square_query_len = 0
        vec = matrix[index]
        for i in range(0, length):
            dq = dq + (vec[i] * vector[i])
            square_query_len = square_query_len + vector[i]**2

        if square_query_len == 0 or document_length[str(index)] == 0:
            #ret.append(0)
            ret[index] = 0
        else:
            #ret.append(dq/(np.sqrt(square_query_len) * document_length[str(index)]))
            ret[index] = dq/(np.sqrt(square_query_len) * document_length[str(index)])
    return ret




def raiseNotImplementedError(param):
    pass



def search_api(query):
    """
    query:[string] 
    return: list of dict, each dict is a paper record of the original dataset
    """
    dict = process_query(query)
    print(dict)

    vector = get_vector(dict)
    matrix = get_matix(dict)
    print(vector)
    # print(vector)
    # for temp in matrix:
    #     print(temp)

    similar = CosSim1(vector, matrix)
    # print(similar)

    pagerank_dict = sorted(similar.items(), key=lambda x: x[1], reverse=True)
    # print(pagerank_dict)

    ret = []
    for i in range(0, 5):
        print("-----------------------------------")
        index, score = pagerank_dict[i]
        ret.append(paper[index])
        print("Document ID: \t", index)
        print("Document title: \t", paper[index]["title"])
        print("Five highest weighted keywords of the document and the corresponding postings lists:")
        temp = document_word_frequency_dict[str(index)]
        word_frequency = sorted(temp.items(), key=lambda x: x[1], reverse=True)
        for j in range(0, 5):
            word, frequency = word_frequency[j]
            # print(word, frequency)
            list = processed_abstract_dict[word][str(index)]
            print(word, "\t ->\t", list)

        if str(index) in unique_word_dict:
            print("The number of unique keywords in the document: \t", unique_word_dict[str(index)])
        else:
            print("The number of unique keywords in the document: \t", "[]")
        print("The magnitude (L2 norm) of the document vector: \t", document_length[str(index)])
        print("The similarity score: \t", score)

    #raise raiseNotImplementedError("You need implement this function")
    return ret


if __name__ == "__main__":
    #search_api("knowledge graph")
    '''
    query = "knowledge graph"
    dict = process_query(query)
    print(dict)

    vector = get_vector(dict)
    matrix = get_matix(dict)
    print(vector)
    # print(vector)
    # for temp in matrix:
    #     print(temp)

    similar = CosSim1(vector, matrix)
    #print(similar)

    pagerank_dict = sorted(similar.items(), key=lambda x: x[1], reverse = True)
    #print(pagerank_dict)

    for i in range(0, 5):
        print("-----------------------------------")
        index, score = pagerank_dict[i]
        print("Document ID: \t", index)
        print("Five highest weighted keywords of the document and the corresponding postings lists:")
        temp = document_word_frequency_dict[str(index)]
        word_frequency = sorted(temp.items(), key=lambda x:x[1], reverse = True)
        for j in range(0, 5):
            word, frequency = word_frequency[j]
            #print(word, frequency)
            list = processed_abstract_dict[word][str(index)]
            print(word,"\t ->\t",list)

        if str(index) in unique_word_dict:
            print("The number of unique keywords in the document: \t", unique_word_dict[str(index)])
        else:
            print("The number of unique keywords in the document: \t", "[]")
        print("The magnitude (L2 norm) of the document vector: \t", document_length[str(index)])
        print("The similarity score: \t", score)
        '''




    '''
    还需要写一个函数，query有没有一起出现在paper的abstract中
    现在得到的similar是分别对knowledg和graph计算得到对值，
    现在确定是否有文件中存在knowledg graph同时出现，
    即查processed_abstract_original_position,
    看是否有paper中postion(knowledg)+1 == postion(graph)
    若有提高权重
    '''