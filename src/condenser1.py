import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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

def condense1(fulltext,n):
    i = 1
    sum = " "
    for sentence in fulltext.split("."):
        if i == 1:
            sentences = sentence
            i = i+1
        elif i==n:
            sentences = sentences + " " + sentence
            i = 1
            sum = sum + " " + summarize1(sentences)
        else:
            sentences = sentences + " " + sentence
            i = i+1
    if i != 1:
        sum = sum + " " + summarize1(sentences)
    return sum

def summarize1(sentences):
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(sentences)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=100,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return summary