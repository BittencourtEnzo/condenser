from collections import defaultdict
from heapq import nlargest
from string import punctuation
import math

import streamlit as st
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize

# text = """Os quatro suspeitos de um ataque terrorista em uma casa de shows perto de Moscou, na Rússia, foram encaminhados a um tribunal distrital neste domingo, 24, denunciados pelo crime. O ataque aconteceu na sexta-feira, 22, e deixou pelo menos 137 mortos. Os homens estavam com graves sinais de tortura, e vídeos nas redes sociais mostram que eles foram espancados durante o interrogatório.
# Segundo informações do jornal The New York Times, os quatro homens são do Tajiquistão, e imigraram para a Rússia a trabalho. Eles foram presos no sábado, 23, e podem pegar prisão perpétua no país.
# De acordo com o tribunal, dois réus se declararam culpados e foram formalmente acusados, sendo eles Dalerjon Mirzoyev e Saidakrami Rachalbalizoda. A declaração dos outros dois suspeitos não foi divulgada, segundo a agência de notícias independente Mediazona.
# O mais jovem dos acusados é Muhammadsobir Fayzov, um barbeiro de 19 anos. Ele foi levado para a sala de audiências diretamente do hospital, em uma cadeira de rodas, e acompanhado por um médico. Ele ficou sentado na cadeira de rodas o tempo todo, na área dos réus, usando um cateter e uma bata hospitalar com parte do corpo exposto. Com ajuda de um tradutor, o suspeito respondeu perguntas em voz baixa e gaguejou, segundo a agência Mediazona.
# Outro suspeito, Rachabalizoda, de 30 anos, tinha um grande curativo no lado direito da cabeça, após ter parte de sua orelha cortada e colocada em sua boca, durante a tortura. A situação foi filmada e publicada na internet.
# A imprensa acompanhou apenas parte das audiências, por preocupação de que detalhes sensíveis sobre a investigação fossem revelados e que os funcionários do tribunal fossem postos em risco de vida. A decisão é comum na Rússia.
# Prisões dos suspeitos de ataque terrorista
# No sábado, 23, foi anunciado pelos Serviços Federais de Segurança da Rússia que 11 suspeitos foram detidos, incluindo os quatro homens que compareceram ao Tribunal. Eles foram presos depois que o carro que usavam para fugir foi interceptado por autoridades perto da fronteira tríplice entre a Rússia, Ucrânia e Bielorrússia, segundo o Times.
# O ataque de sexta-feira aconteceu quando quatro homens armados abriram fogo dentro de um teatro, contra o público de um show de rock, do grupo russo Piknik. O show ainda não tinha começado.
# Foram usados explosivos que causaram um incêndio no prédio, cujo teto desabou sobre as vítimas. Além dos mortos, cerca de 182 pessoas ficaram feridas e mais de 100 estão hospitalizadas.
# Autoria do crime
# O presidente Vladimir Putin sugeriu que, por os suspeitos terem sido detidos na rodovia que leva à Ucrânia, o crime estava ligado à guerra com o país vizinho. No entanto, os Estados Unidos já haviam alertado sobre um possível ataque terrorista com autoria de uma facção da organização jihadista extremista Estado Islâmico, conhecida como ISIS-K (ou Estado Islâmico de Khorasan), que reivindicou a responsabilidade.
# O primeiro suspeito de terrorismo, Mirzoyev, de 32 anos, apareceu no Tribunal com um olho roxo, hematomas e cortes pelo rosto. Ele ficou apoiado na parede de vidro da sala durante a leitura da acusação. Ele tem 4 filhos e tinha permissão para morar temporariamente em Novosibirsk, do sul da Sibéria. Segundo relatórios, o documento estava vencido.
# Já Rachabalizoda é casado e pai de um filho. Ele disse que estava registrado para trabalhar na Rússia legalmente, mas não se lembrava exatamente onde, segundo o jornal.
# O quarto suspeito, Shamsidin Fariduni, de 25 anos, é casado e tem um bebê de 8 meses. Ele trabalhava em uma fábrica em Podolsk, sudoeste de Moscou. O homem também havia trabalhado como faz-tudo em Krasnogorsk, no subúrbio de Moscou onde ocorreu o ataque na casa de shows.
# O Estado Islâmico tem usado migrantes da Ásia Central que vão para a Rússia em busca de emprego para seus ataques. Os adeptos costumam ser pessoas revoltadas com a discriminação que enfrentam no país.
# """

stopwords_ptbr = set(stopwords.words('portuguese')+list(punctuation))
def remove_stop_words_and_punct_in_portuguese(text):
    words= word_tokenize(text.lower())
    return [word for word in words if word not in stopwords_ptbr]

def summarize_text_portuguese(text, n_sent=2):
    words_not_stopwords = remove_stop_words_and_punct_in_portuguese(text)
    sentences = sent_tokenize(text)
    frequency = FreqDist(words_not_stopwords)
    important_sentences = defaultdict(int)
    
    for i, sentence in enumerate(sentences):
        for word in word_tokenize(sentence.lower()):
            if word in frequency:
                important_sentences[i] += frequency[word]
                
    numb_sent = n_sent
    idx_important_sentences = nlargest(numb_sent,
                                       important_sentences,
                                       important_sentences.get)
    
    sum = ' '
    for i in sorted(idx_important_sentences):
        sum = sum + sentences[i]+" "
    return sum

def condense5(fulltext):
    k = 0
    for sentence in fulltext.split("."):
        k = k+1
    return summarize_text_portuguese(fulltext,math.floor(k))