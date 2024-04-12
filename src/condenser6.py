from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
import math

def condense6(fulltext):
    k = 0
    for sentence in fulltext.split("."):
        k = k+1
    parser = PlaintextParser.from_string(fulltext, Tokenizer('portuguese'))
    summarizer = LuhnSummarizer()
    resumo = summarizer(parser.document, k)

    sum = ""
    for sentence in resumo:
        sum = sum + str(sentence)
    return sum