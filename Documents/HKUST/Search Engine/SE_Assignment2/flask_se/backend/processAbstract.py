import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import math
import json

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
#proccessAbstrctDict = {}

def test():
    with open("paper.json", "r") as f:
        paperlist = json.loads(f.read())
    abstract = paperlist[1]["abstract"]
    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(abstract)
    print(word_tokens[47])
    


#show the position of the word in the paper's index(with out stoping word and marks)
#struct:
#{"wordname":{
#               "paperindex": [a list of position]
#             }
def process_abstract():
    proccessAbstrctDict = {}
    with open("paper.json", "r") as f:
        paperlist = json.loads(f.read())

    index = -1
    for dic in paperlist:
        #the index of paper, value from 0 to 777
        index = index + 1

        #get the abstract
        abstract = dic['abstract']

        # Discard punctuation marks & perform tokenization
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(abstract)

        # Stop word remove & stemming
        stem_abstract = []
        for word in word_tokens:
            if word not in stop_words:
                stem_abstract.append(ps.stem(word))

        # n is the nth word in the stem_abstract of the No.index paper
        n = 0
        for word in stem_abstract:
            if word in proccessAbstrctDict:  # No.n word exist in the dict
                if index in proccessAbstrctDict[word]:# index of current paper exist in the dict[word]
                    list = proccessAbstrctDict[word][index]
                    list.append(n)
                    n = n + 1
                else:# index of current paper do not exist in the dict[word]
                    proccessAbstrctDict.setdefault(word, {})[index] = [n]
                    n = n + 1

            else:# No.n word do not exist in the dict
                proccessAbstrctDict.setdefault(word, {})[index] = [n]
                n = n + 1

    with open("processed_abstract.json", "w") as f:
        json.dump(proccessAbstrctDict, f)

def process_abstract_original_position():
    proccessAbstrctDict = {}
    with open("paper.json", "r") as f:
        paperlist = json.loads(f.read())

    index = -1
    for dic in paperlist:
        #the index of paper, value from 0 to 777
        index = index + 1

        #get the abstract
        abstract = dic['abstract']

        # Discard punctuation marks & perform tokenization
        tokenizer = RegexpTokenizer(r'\w+')
        word_tokens = tokenizer.tokenize(abstract)

        # Stemming
        stem_abstract = []
        for word in word_tokens:
            stem_abstract.append(ps.stem(word))

        # n is the nth word in the stem_abstract of the No.index paper
        n = 0
        for word in stem_abstract:
            if word not in stop_words:
                if word in proccessAbstrctDict:  # No.n word exist in the dict
                    if index in proccessAbstrctDict[word]:# index of current paper exist in the dict[word]
                        list = proccessAbstrctDict[word][index]
                        list.append(n)
                        n = n + 1
                    else:# index of current paper do not exist in the dict[word]
                        proccessAbstrctDict.setdefault(word, {})[index] = [n]
                        n = n + 1

                else:# No.n word do not exist in the dict
                    proccessAbstrctDict.setdefault(word, {})[index] = [n]
                    n = n + 1
            else:
                n = n+1

    with open("processed_abstract_original_position.json", "w") as f:
        json.dump(proccessAbstrctDict, f)


#caculate the frequency of every word in a document
#struct:
#{"wordname":{
#               "paperindex": frequency of the word in this paper's abstract
#             }
def build_word_document_frequecy():
    word_document_frequency_dict = {}
    with open("processed_abstract.json", "r") as f:
        wordlist = json.loads(f.read())

    for word in wordlist:
        for index in wordlist[word]:
            frequency = len(wordlist[word][index])
            word_document_frequency_dict.setdefault(word, {})[index] = frequency

    with open("word_document_frequency.json", "w") as f:
        json.dump(word_document_frequency_dict, f)


#caculate the frequency of every word in a document
#struct:
#{"paperindex":{
#               "wordname": frequency of the word in this paper's abstract
#             }
#}
def build_document_word_frequency():
    document_word_frequency_dict = {}
    with open("word_document_frequency.json", "r") as f:
        wordlist = json.loads(f.read())


    for index in range(0, 778):
        for word in wordlist:
            if str(index) in wordlist[word]:
                document_word_frequency_dict.setdefault(index, {})[word] = wordlist[word][str(index)]

    #print(document_word_frequency_dict)
    with open("document_word_frequency.json", "w") as f:
        json.dump(document_word_frequency_dict, f)

def build_df_idf():
    df_dict = {}
    idf_dict = {}
    with open("word_document_frequency.json", "r") as f:
        wordlist = json.loads(f.read())

    for word in wordlist:
        df_dict[word] = len(wordlist[word])

    with open("df.json", "w") as f:
        json.dump(df_dict, f)

    for word in df_dict:
        idf_dict[word] = math.log(778/df_dict[word], 2)

    with open("idf.json", "w") as f:
        json.dump(idf_dict, f)

def build_l2norm():
    l2norm_dict = {}
    with open("document_word_frequency.json", "r") as f:
        tf_dict = json.loads(f.read())
    with open("idf.json", "r") as f:
        idf_dict = json.loads(f.read())

    for index in tf_dict:
        temp_dict = tf_dict[index]
        temp = 0
        for word in temp_dict:
            temp = temp + (temp_dict[word]*idf_dict[word])**2

        l2norm_dict[index] = math.sqrt(temp)

    with open("Document_length.json", "w") as f:
        json.dump(l2norm_dict, f)

def build_unique_word():
    unique_dict = {}
    with open("df.json", "r") as f:
        df_dict = json.loads(f.read())
    with open("word_document_frequency.json", "r") as f:
        word_document_frequecy_dict = json.loads(f.read())

    for word in df_dict:
        if df_dict[word] == 1:
            temp = word_document_frequecy_dict[word]
            for index in temp:
                print(word, index)
                if index not in unique_dict:
                    list = [word]
                    unique_dict[index] = list
                else:
                    unique_dict[index].append(word)

    with open("unique_word.json", "w") as f:
        json.dump(unique_dict, f)






if __name__ == '__main__':
    # process_abstract()
    # build_word_document_frequecy()
    # build_document_word_frequency()
    #process_abstract_original_position()
    #test()
    #build_df_idf()
    #build_l2norm()
    build_unique_word()