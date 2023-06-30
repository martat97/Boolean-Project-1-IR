#Libraries
import pandas as pd
import random
import numpy as np
import BTree as bt
import time
import string
import sys
import os
import nltk
#nltk.download('punkt') #automatically download of the package
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
pd.options.mode.chained_assignment = None  # default='warn'

#Class of the system with all the functions
class BRetrievalSystem:
    #Initiliaze all the variables of the system
    def __init__(self, input_data, input_column, input_length):
        self.input_data = input_data.head(input_length) #Input dataset
        self.document_column = input_column #Input column of the dataset
        self.stopwords = self.define_stopwords() #List of stopwords
        self.data = self.preprocessing() #Dataset preprocessed (that will be used for the index)
        self.inv_index_pos = self.inv_index_pos_builder() #Positional Inverted Index
        self.btree = self.btree_builder() #Btree to store the terms

    #

    #Define the stopwords from ntlk libraries (english language)
    def define_stopwords(self):
        stopwords = nltk.corpus.stopwords.words('english')
        return stopwords

    #

    #Preprocessing function
    def preprocessing(self):
        print('Preprocessing text...')
        data = self.input_data
        textcolumnname = self.document_column

        #1 Punctuation Removal
        #defining the function to remove punctuation
        string.punctuation
        def remove_punctuation(text):
            punctuationfree="".join([i for i in text if i not in string.punctuation])
            return punctuationfree
        #storing the puntuaction free text
        data['clean_msg']= data[textcolumnname].apply(lambda x:remove_punctuation(x))

        #2 Lowering the text
        data['msg_lower']= data['clean_msg'].apply(lambda x: x.lower())


        #3 Tokenization
        for i in range(0,len(data)):
            data["msg_lower"][i] = nltk.word_tokenize(data["msg_lower"][i])

        #4 Stopwords
        stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords = stopwords
        #defining the function to remove stopwords from tokenized text
        def remove_stopwords(text):
            output= [i for i in text if i not in self.stopwords]
            return output
        #applying the function
        data['no_stopwords']= data['msg_lower'].apply(lambda x:remove_stopwords(x))

        #5 Stemming
        #importing the Stemming function from nltk library
        from nltk.stem.porter import PorterStemmer
        #defining the object for stemming
        porter_stemmer = PorterStemmer()

        #defining a function for stemming
        def stemming(text):
            stem_text = [porter_stemmer.stem(word) for word in text]
            return stem_text

        data['msg_stemmed']=data['no_stopwords'].apply(lambda x: stemming(x))

        data = data['msg_stemmed']
        return data

    #

    #Building the Positional Inverted Index
    def inv_index_pos_builder(self):
        print('Building Positional Inverted Index...')

        #1 BUILDING INVERTED INDEX

        doc_terms = pd.DataFrame({"terms": self.data})

        #We tag each document with the corresponding DocID
        doc_terms["docId"] = doc_terms.index.astype(str)

        #For each document we extract the sequence of terms putting each one of them in a new row
        terms = doc_terms.explode("terms")

        # Group equal terms to obtain a list of documents for each term
        inv_index = terms.groupby("terms")["docId"].apply(list).reset_index()
        inv_index = inv_index.rename(columns={"terms": "term"})


        #2 BUILDING POSITIONAL INVERTED INDEX
        inv_index_pos = []

        #For each term in the inverted index (row of it), for each document of the term, I get the position by comparing with
        #the terms of the document. When the term is equal, I add the position
        for row in range(len(inv_index)):
            term = inv_index['term'][row]
            docIds = []
            positions = []

            for doc in inv_index['docId'][row]:
                pos_list = []
                for index_word in range(len(self.data[int(doc)])):
                    if inv_index['term'][row] == self.data[int(doc)][index_word]:
                        pos_list.append(index_word)

                if pos_list:
                    docIds.append(doc)
                    positions.append(pos_list)

            # Remove some possible duplicates that may occur
            if docIds:
                unique_docs = []
                unique_positions = []
                seen = set()
                for doc, pos in zip(docIds, positions):
                    doc_tuple = (doc, tuple(pos))
                    if doc_tuple not in seen:
                        seen.add(doc_tuple)
                        unique_docs.append(doc)
                        unique_positions.append(pos)

                inv_index_pos.append({'term': term, 'docId': [[doc, pos] for doc, pos in zip(unique_docs, unique_positions)]})

        inv_index_pos = pd.DataFrame(inv_index_pos)


        #3 ADD ROTATIONS OF THE TERMS
        #I define the "rotation" function, here the string new_el is updated on each iteration in which I shift one position
        #of the word at each step
        def rotation(term):
            rotations = []
            new_el = term+'$'
            for i in range(0,len(new_el)):
                rotations.append(new_el)
                new_el = new_el[1:len(new_el)] + new_el[0]
            return rotations

        #I apply the function "rotation" to each term, and add the result in a new column called "rotations"
        inv_index_pos['rotations'] = inv_index_pos['term'].apply(rotation)

        #The index is saved on the disk automatically
        os.makedirs('index_output', exist_ok=True)
        inv_index_pos.to_csv('index_output/pos_inverted_index.csv', index=False)

        return inv_index_pos

    #

    #Manually Save of the Index in a .csv file on disk
    def index_save(self):
        #I save the index in the disk
        os.makedirs('index_output', exist_ok=True)
        self.inv_index_pos.to_csv('index_output/pos_inverted_index.csv', index=False)


    #

    #BUILDING BTREE
    #I store each row of the inverted index in a node of a Btree to improve efficiency on searching
    def btree_builder(self):
        print('Building BTree...')

        print(str(len(self.inv_index_pos)) + ' nodes in the tree')
        #t is selected in proportion to the number of nodes
        #n = number of keys
        #n = 2*t-1
        #t = (n+1)/2
        #for now, I set t = 32
        t = 32
        B = bt.BTree(t)
        keys = self.inv_index_pos

        for i in range(0,len(keys)):
            B.insert(self.inv_index_pos.iloc[i], 'term')

        return B

    #

    ### Search possible results of the query! ###
    def print_result(self, inp_query):

        # All the functions defined inside this method:

        ### Conversion of the query, from string to the 3 vectors ###
        def conversion_query(inp_query, query, op_query, op_not):
            word = ''
            is_not = 0
            arraywords = []
            for i in range(0,len(inp_query)):
                if (inp_query[i] != ' '):
                    if (inp_query[i] == '&'):
                        op_query.append('AND')
                    elif (inp_query[i] == '|'):
                        op_query.append('OR')
                    elif (inp_query[i] == '^'):
                        is_not = 1
                    else: # i == letter of word
                        word+= inp_query[i]
                        if (i == len(inp_query)-1): #last letter of the inp
                            arraywords.append(word)
                            query.append(arraywords)
                            op_not.append(is_not)

                else: #i == space
                    if (word != ''):
                        arraywords.append(word)
                        if ((inp_query[i+1] in ['&','|','^']) | (i+1 == len(inp_query))):
                            query.append(arraywords)
                            op_not.append(is_not)
                            is_not = 0
                            arraywords = []
                        word = ''

        ### Boolean Operations ###
        def and_function(list1, list2):
            result = []
            i = 0
            j = 0
            while(i < len(list1) and j < len(list2)) :
                if (list1[i] == list2[j]):
                    result.append(list1[i])
                    i+=1
                    j+=1
                else:
                    if (list1[i] > list2[j]):
                        j+=1
                    else:
                        i+=1
            return result


        def or_function(list1, list2):
            result = []
            i = 0
            j = 0
            while(i < len(list1) and j < len(list2)) :
                if (list1[i] == list2[j]):
                    result.append(list1[i])
                    i+=1
                    j+=1
                else:
                    if (list1[i] > list2[j]):
                        result.append(list2[j])
                        j+=1
                    else:
                        result.append(list1[i])
                        i+=1
            result = result + list1[i:len(list1)]
            result = result + list2[j:len(list2)]
            return result

        def not_function(list1):
            result = []
            for i in range(0, len(self.data)):
                if (str(i) not in list1):
                    result.append(str(i))
            return result


        def and_function_pos(list1, list2):
            result = []
            i = 0
            j = 0
            while(i < len(list1) and j < len(list2)) :
                if (list1[i][0] == list2[j][0]):
                    resultpos = []
                    k = 0
                    z = 0
                    #scan the positions
                    while(k < len(list1[i][1]) and z < len(list2[j][1])) :
                        #consecutive positions!
                        if (list1[i][1][k]+1 == list2[j][1][z]):
                            resultpos.append(list1[i][1][k]+1)
                            k+=1
                            z+=1
                        else:
                            if (list1[i][1][k] > list2[j][1][z]):
                                z+=1
                            else:
                                k+=1

                    if (len(resultpos) > 0):
                        result.append( [list1[i][0],resultpos] )
                    i+=1
                    j+=1
                    resultpos = []
                else:
                    if (int(list1[i][0]) > int(list2[j][0])):
                        j+=1
                    else:
                        i+=1
            return result



        ###Wildcard Operations###


        #Check if a term is wildcard
        def is_wildcard(term):
            i = 0
            result = False
            while (i < len(term) ):
                if (term[i] == '*'):
                    result = True
                    break #stop while
                i+=1
            return result

        # From a set of rotations of a term, get the one with '*' at the end
        def get_ending_wildcard(rotations):
            i = 0
            ending_wildcard = ''
            while (i < len(rotations) ):
                if (rotations[i][len(rotations[i])-1] == '*'):
                    ending_wildcard = rotations[i]
                    break
                i+=1
            return ending_wildcard

        # it returns true if there exists a wildcard with '*' at the end matching the ending wildcard
        def matching_term_wildcard(rotations, ending_wildcard):
            i = 0
            result = False
            ending_wildcard = ending_wildcard[0:len(ending_wildcard)-1]
            while (i < len(rotations) ):
                if (len(rotations[i])>=len(ending_wildcard)):
                    if (rotations[i][0:len(ending_wildcard)] == ending_wildcard):
                        result = True
                        break
                i+=1
            return result

        # it returns True if there exists a rotation that is greater than the ending wildcard (matching)
        def greater_term_wildcard(rotations, ending_wildcard):
            i = 0
            result = False
            ending_wildcard = ending_wildcard[0:len(ending_wildcard)-1]
            dollar_pos = 0
            if(ending_wildcard[0] != '$'):
                dollar_pos = len(ending_wildcard)-1

            j = len(rotations)-1
            while (i < len(rotations)):
                if (len(rotations[i])>=len(ending_wildcard)):
                    if (j == dollar_pos):
                        if (rotations[i][0:len(ending_wildcard)] > ending_wildcard):
                            result = True
                            break
                else:
                    if (dollar_pos == 0):
                        ending_wildcard = ending_wildcard[1:len(ending_wildcard)] #remove the starting '$'!
                    if (rotations[i] > ending_wildcard):
                        result = True
                        break
                i+=1
                j-=1
            return result


        #Is just a single wildcard? I count the number of '*'
        def is_single_wildcard(wildcard):
            count_star = 0
            for i in wildcard:
                if (i == '*'):
                    count_star+=1

            return (count_star==1)


        #It applies the concept of "String Matching"
        #Scanning of both term and wildcard.
        #It check if each "piece" between the '*' is matching between the two.
        #It stores in a bit vector the matching of the 'pieces', 1=matching, 0=not matching
        def matching_generic_wildcard(term, wildcard):
            i = 0  #term
            j = 0  #wildcard
            len_piece_wildcard = 0
            vector_piece_wildcard = []

            while (i < len(term) and j < len(wildcard)):
                # I have to find from the term the piece of wildcard
                if (wildcard[j] == '*'):
                    j+=1
                    if (j == len(wildcard)):
                        break
                    piece_wildcard = ''
                    while (j < len(wildcard) and wildcard[j] != '*'):
                        piece_wildcard = piece_wildcard + wildcard[j]
                        j+=1
                    len_piece_wildcard+=1
                    while(len(term)-i >= len(piece_wildcard)):
                        if (term[i:i+len(piece_wildcard)] == piece_wildcard):
                            if (j != len(wildcard)):
                                vector_piece_wildcard.append(1)
                                i = i+len(piece_wildcard)
                            else: #last piece of wildcard, it must end with this
                                if (i+len(piece_wildcard) == len(term) ): #I'm at the end!
                                    vector_piece_wildcard.append(1)
                                    i = i+len(piece_wildcard)
                                else:
                                    i = i+1
                                    len_piece_wildcard = len_piece_wildcard - 1
                                    j = j-len(piece_wildcard)-1
                            break
                        i+=1
                else: #case not '*' at the beginning
                    piece_wildcard = ''
                    while (j < len(wildcard) and wildcard[j] != '*'):
                        piece_wildcard = piece_wildcard + wildcard[j]
                        j+=1
                    len_piece_wildcard+=1
                    if(len(term)-i >= len(piece_wildcard)):
                        if (term[i:i+len(piece_wildcard)] == piece_wildcard):
                            vector_piece_wildcard.append(1)
                            i = i+len(piece_wildcard)
                        else:
                            break

            return (sum(vector_piece_wildcard) == len_piece_wildcard)

        #Rotations of the term
        def rotation(term):
            rotations = []
            new_el = term+'$'
            for i in range(0,len(new_el)):
                rotations.append(new_el)
                new_el = new_el[1:len(new_el)] + new_el[0]
            return rotations


        ###Search of an element###
        def search_element(query_element):
            actual_term_phrase = []
            # For wildcard cases:
            if (is_wildcard(query[i][j])):
                # For single wildcard (I can reduce the complexity)
                if (is_single_wildcard(query_element)):
                    word_rotations = rotation(query_element)
                    ending_wildcard = get_ending_wildcard(word_rotations)

                    actual_term_phrase = pd.DataFrame(columns=['term','docId','rotations'])
                    # Do I scan all the tree?
                    if (query_element[len(query_element)-1] == '*'): #trailing wildcard
                        actual_term_phrase = self.btree.scan_tree(self.btree.root, 'rotations', matching_term_wildcard,greater_term_wildcard,ending_wildcard,actual_term_phrase)
                    else:#other single wildcards?
                        actual_term_phrase = self.btree.scan_all_tree(self.btree.root, 'rotations', matching_term_wildcard,ending_wildcard,actual_term_phrase)

                else:
                    # For multiple wildcards, I apply the matching generic wildcard for each node
                    actual_term_phrase = pd.DataFrame(columns=['term','docId','rotations'])
                    actual_term_phrase = self.btree.scan_all_tree(self.btree.root, 'term', matching_generic_wildcard,query_element,actual_term_phrase)

                    # Reset the indexes
                actual_term_phrase.reset_index(drop=True)
                print()
                print('Terms matching the wildcard ' + query_element)
                print(list(actual_term_phrase['term']))
                print()

            # Not wildcard case, I simply search the term
            else:
                #select the term from the dictionary if it exists
                actual_term_phrase = pd.DataFrame(columns=['term','docId','rotations'])
                actual_term_phrase = self.btree.search_key(query_element, actual_term_phrase)

            return actual_term_phrase



        ###Spelling Correction###
        def edit_distance(str1, str2):
            m = len(str1)
            n = len(str2)

            # Creation of the matrix (m+1) x (n+1)
            dp = [[0] * (n+1) for _ in range(m + 1)]

            # Initialize first column and row
            for i in range(m+1):
                dp[i][0] = i
            for j in range(n+1):
                dp[0][j] = j

            # Compute edit distance applying the rules
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if str1[i - 1] == str2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = min(dp[i - 1][j] + 1,      # Delete
                                       dp[i][j - 1] + 1,      # Insert
                                       dp[i - 1][j - 1] + 1)  # Replace

            return dp[m][n]


        #Search of the closest term of a query element, with min edit distance and high frequency
        def search_closest(query_element):
            closest_term = ''
            edit_distances = []
            for term in self.inv_index_pos['term']:
                #save results in a column
                new_edit_distance = edit_distance(term, query_element)
                edit_distances.append(new_edit_distance)
            #max values
            min_edit_distance = min(edit_distances)
            min_edit_vector = []
            for element_pos in range(0,len(edit_distances)):
                if (edit_distances[element_pos] == min_edit_distance):
                    min_edit_vector.append(element_pos)

            #max_edit_vector is a vector with positions of term with max edit distance
            #I take the term with high absolute frequency in documents
            max_frequency = 0
            for element_pos in min_edit_vector:
                row = self.inv_index_pos.iloc[element_pos] #same complexity as scanning btree
                frequency = 0
                for doc in row['docId']:
                    frequency = frequency + (int(doc[0])*len(doc[1]))

                if (frequency > max_frequency):
                    max_frequency = frequency
                    closest_term = row['term']

            return closest_term




        ###Functions defined, now I use them to Search possible Results###

        #Define query vectors
        query = []
        op_query = []
        op_not = []
        conversion_query(inp_query, query, op_query, op_not)

        #Final result (list of docIds)
        result = []

        #Iteration for each element of the query separated with &, | (can be a single term, phrase or wildcards)
        for i in range(0,len(query)):
            new_query = [] #The new query without the stop words I remove from the query

            #remove stop words from query
            for j in range(0,len(query[i])):
                if (query[i][j] not in self.stopwords):
                    new_query.append(query[i][j])
                else:
                    print('The term "' + query[i][j] + '" is a stopword, it is removed from the query')

            query[i] = new_query

            #Final result for a phrase, list of docIds with positions
            #result_phrase is the previous result I take into account for each element of a phrase
            result_phrase = []

            #Iteration for each "piece" of the query (can be a term or a wildcard inside a phrase)
            for j in range(0,len(query[i])):
                #actual_phrase is the actual result I take into account
                actual_phrase = []

                #search the terms matching my single term/wildcard
                actual_term_phrase = search_element(query[i][j])

                #Term not matching anything, Spelling Correction (not in wildcard cases)
                if (len(actual_term_phrase)==0  and is_wildcard(query[i][j]) == False):
                    print('No results found for term "' + query[i][j] + '"')

                    if (query[i][j] in self.stopwords):
                        print('The term "' + query[i][j] + '" is a stopword')
                        print('')
                    else:
                        closest_term = search_closest(query[i][j]) #search the closest term with edit distance
                        print('Did you mean "' + closest_term + '"?')
                        print('')
                        result = []
                        break

                        #Term/terms matching something, I compare them with the previous terms in the phrase,
                #to see if the positions are adjacent
                #(remember that with wildcards I can obtain multiple terms)
                else:
                    #Column to list
                    actual_phrase = actual_term_phrase['docId'].tolist()
                    if (j == 0):
                        #initial case: I save the Actual Result into the final Result (I will compare it with the next term/terms)
                        result_phrase = actual_phrase
                    else:
                        #Next case, I have to compare the documents and positions between the terms/wildcards in the two sets:
                        #Result_phrase for the previous result saved before (j-1)
                        #Actual_phrase for the actual result (j)

                        #Local list in which I save the result
                        new_actual_phrase = []
                        #Iterations of elements found (in case of wildcard there can be multiple terms found in the phrase)
                        for r in result_phrase:
                            for s in actual_phrase:
                                partial_result = and_function_pos(r,s) #apply and function checking consecutive positions
                                if (len(partial_result)!=0): #terms matching!
                                    for el in partial_result:
                                        new_actual_phrase.append([el])
                                        #Update the result
                        result_phrase = new_actual_phrase

                        #Last element of the phrase, I have to remove the additional external list for each element
                        #(it happened beacuse I had two big lists of different sets of doc/position to compare in
                        # the two previous iterations)
                        if (j == len(query[i])-1):
                            corr_result_phrase = []
                            for element in result_phrase:
                                corr_result_phrase.append(element[0]) #Remove the additional external list
                            result_phrase = corr_result_phrase #Correct!

            #Same as before:
            #Actual list, for the actual documents matching the phrase
            #Final result, for the previous documents matching that I'll compare with
            actual_result = []

            #Check to the type of list I have as a result. I expect:
            #[ [ID, [positions]], [ID2, [positions]], etc...]

            #One single term: I had an additional external list to "remove"
            if (len(query[i]) == 1):
                for el in result_phrase:
                    for inside_el in el: #I have to go more inside it
                        actual_result.append(inside_el[0])
                        #No additional external list, it is already correct
            else:
                for el in result_phrase:
                    actual_result.append(el[0])

                    #Remove eventual duplicates from actual_result and Sort it (to be sure and avoid possible exceptions)
            actual_result[:] = [x for i, x in enumerate(actual_result) if i == actual_result.index(x)]
            actual_result.sort()

            #Handle different Boolean Operations

            #Not case
            if (op_not[i] == 1):
                actual_result = not_function(actual_result)

            #As in previous case, in the first iteration I save the actual result in the final result
            if (i == 0):
                result = actual_result
            else:
                #Next elements, I compare the documents matching the previous case
                #And/Or cases
                if (op_query[i-1] == 'AND'):
                    result = and_function(result,actual_result)

                if (op_query[i-1] == 'OR'):
                    result = or_function(result,actual_result)


                    #I print the final document matching
        print("Documents matching the query:")
        print(result) #I print the list of documents

        if (len(result) == 0): #No results matching
            print('None')
        else: #Some results matching, I print the Id and text of the document
            for el in result:
                print((int(el), self.input_data[self.document_column][int(el)]))




