import sys
sys.path.append('../')
from src.condenser1 import condense1
from src.condenser2 import condense2
from src.condenser3 import condense3
from src.condenser4 import condense4
from src.condenser5 import condense5
from src.condenser6 import condense6
import pandas as pd
import re
from langchain_community.document_loaders import UnstructuredFileLoader#, RecursiveCharacterTextSplitter
import time


loader = UnstructuredFileLoader("../data/CEABolsaProtegida.txt")
docs = loader.load()
fulltext = docs[0].page_content

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
fulltext = WHITESPACE_HANDLER(fulltext)
# n = 4 #numero de frases por chunk
# i = 1
# sum1 = " "
# sum2 = " "
# sum3 = " "
# sum4 = " "
# for sentence in text.split("."):
#     if i == 1:
#         sentences = sentence
#         i = i+1
#     elif i==n:
#         sentences = sentences + " " + sentence
#         i = 1
#         sum1 = sum1 + " " + condense1(sentences)
#         sum2 = sum2 + " " + condense2(sentences)
#         sum3 = sum3 + " " + condense3(sentences)
#         sum4 = sum4 + " " + condense4(sentences)
#     else:
#         sentences = sentences + " " + sentence
#         i = i+1

# if i != 1:
#     sum1 = sum1 + " " + condense1(sentences)
#     sum2 = sum2 + " " + condense2(sentences)
#     sum3 = sum3 + " " + condense3(sentences)
#     sum4 = sum4 + " " + condense4(sentences)
n = 4 #tamanho do chunk

start = time.time()
sum1 = condense1(fulltext,n)
end = time.time()
tempo1 = end - start

start = time.time()
sum2 = condense2(fulltext,n)
end = time.time()
tempo2 = end-start

start = time.time()
sum3 = condense3(fulltext,n)
end = time.time()
tempo3 = end-start

start = time.time()
sum4 = condense4(fulltext,n)
end = time.time()
tempo4 = end-start

start = time.time()
sum5 = condense5(fulltext) 
end = time.time()
tempo5 = end-start

start = time.time()
sum6 = condense6(fulltext)
end = time.time()
tempo6 = end-start

print(tempo1,"\n",tempo2,"\n",tempo3,"\n",tempo4,"\n",tempo5,"\n",tempo6)

with open("CEABolsaProtegida.txt", "w") as text_file:
    text_file.write("\n\n Primeiro resumo: "+sum1+ "\n Segundo resumo: "+sum2+ "\n Terceiro resumo: "+sum3+ "\n Quarto resumo: "+sum4+ "\n Quinto resumo: "+sum5+ "\n Sexto resumo: "+sum6)
# print("Primeiro resumo: ",sum1,"\n\n")
# print("Segundo resumo: ",sum2,"\n\n")
# print("Terceiro resumo: ",sum3,"\n\n")
# print("Quarto resumo: ",sum4,"\n\n")
# print("Quinto resumo: ",sum5,"\n\n")