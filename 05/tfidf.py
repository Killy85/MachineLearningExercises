from sklearn.feature_extraction.text import TfidfVectorizer
import os
import math
import operator

#This is where we store all the documents
all_documents = []
all_documents_name = []

#This is the path to the training corpus
url ='./SimpleText/SimpleText_train/'

#We load all the documents of the corpus in memory
for file in os.listdir(url):
    all_documents.append(open(url + file,'r+').read())
    all_documents_name.append(file)


#This is the path we will use for test documents
test_path = './SimpleText/SimpleText_test/'
#Store all the path for files 
test_documents = [test_path +elem for elem in os.listdir(test_path)]

# This function compute the cosine similarity for a matrix
# and a sparse matrix ( the vector 2 must be a sparse matrix)
def cosine_similarity(vector1, vector2):
    dot_product = vector2.multiply(vector1).sum()
    magn1 = math.sqrt(sum([val**2 for val in vector1]))
    magn2 = math.sqrt(vector2.power(2).sum())
    magnitude =  magn1 * magn2
    if not magnitude:
        return 0
    return dot_product/magnitude
# We create the vectorizer
vectorizer = TfidfVectorizer()
# And use it to fit/transforme our whole train set
X = vectorizer.fit_transform(all_documents)
# In this dict we store data about our recommendations
recommendation_for_paper = {}
for count_0, doc_0 in enumerate(X.toarray()):
    for elem in test_documents:
        # We create the sparse matrix for the document we want to compare
        doc_comp = vectorizer.transform([open(elem,'r+').read()])
        try:
            recommendation_for_paper[elem][count_0] = cosine_similarity(doc_0, doc_comp)
        except KeyError :
            recommendation_for_paper[elem] = {}
            recommendation_for_paper[elem][count_0] = cosine_similarity(doc_0, doc_comp)

for key,value in recommendation_for_paper.items():
    # For each tests document, we display the 5 most relevant papers
    print(f'The file {key} is considered linked to files :')
    for elem in sorted(value.items(), key=operator.itemgetter(1),reverse=True)[:5]:
        print(all_documents_name[elem[0]] + " with a score of " + str(elem[1]))
    