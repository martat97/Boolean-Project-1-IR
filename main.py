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
    data = pd.read_csv('dataset/news_summary.csv', encoding='latin')
    input_column = 'text'

    while True:
        try:
            brs = bre.BRetrievalSystem(data, input_column, 1000) #default length
            break
        except:
            print('Unable to create the IR system properly')

    print()
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



