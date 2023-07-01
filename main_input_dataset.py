#Libraries
import pandas as pd
import numpy as np
import BTree as bt
import BRetrieval as bre
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
pd.options.mode.chained_assignment = None  # default='warn'

def main():
    print('Welcome to the Boolean Retrieval System!')
    print()
    print('Datasets accepted with .csv format. You have to specify the name of the column containing the documents. String format accepted')
    # Handling exceptions with csv file
    encodings = ['utf-8','latin']
    enc_pos = 0
    while True:
        try:
            dataset_location = input('Dataset location:')
            data = pd.read_csv(dataset_location, encoding=encodings[enc_pos])
            break
        except FileNotFoundError:
            print("File not found")
        except UnicodeDecodeError:
            print("Unable to parse the file with the encoding " + encodings[enc_pos])
            if (enc_pos < len(encodings)):
                enc_pos+=1
                print("Encoding changed to " + encodings[enc_pos])
            else:
                print("Unable to encode the file with any encoding")
                enc_pos = 0
        #except EmptyDataError:
        #    print("Empty file")
        print("Try again")

    #data = pd.read_csv("news_summary.csv", encoding='latin')



    # Just print the first 20 columns (we don't want to print all of them if there are a lot)
    columns = list(data.columns)
    column_limit = 20
    print('Columns found in the file: ')
    print(columns[0:column_limit])
    if (len(columns)>column_limit):
        print("... and other " + str(len(columns)-column_limit) +" columns")

    # Handling exceptions with input column
    # Check if it is string?
    while True:
        input_column = input('Type the Name of the Column containing the documents:')
        if (input_column not in columns):
            print("Column not found in the document. Try again")
        else:
            #print(type(data[input_column]))
            break

    #input_column = 'text'

    while True:
        try:
            brs = bre.BRetrievalSystem(data, input_column, 1000) #default length chosen
            break
        except:
            print('Unable to create the IR system properly')
            break

    print('Boolean Operators format: and = "&" or = "|" not = "^"')
    print('Query format: term1 booloperator term2 booloperator term3...')
    print('Example: hello & my name | is & ^Paolo')


    # -stop- is the combination to exit the program
    while True:
        print('You can write a query in respect to the format accepted')
        print('You can write: "-exit-" to exit the program, "-save-" to save the index on disk')
        query = input('Input your Query:')
        #*error of querys to handle!
        # query errors:
        # spaces?
        # symbols (&&& not valid or different ones with $)
        if (query == '-exit-'):
            break
        elif (query == '-save-'):
            brs.index_save()
            print('Index saved successfully in location: index_output/pos_inverted_index.csv')
            print()
        else:
            print()
            print('The system is searching..')
            brs.print_result(query)
            print()

if __name__ == '__main__':
    main()






