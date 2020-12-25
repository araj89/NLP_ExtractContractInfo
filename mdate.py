import spacy
import pickle
from spacy.lang.en import English
import numpy as np


date_classes = {
    'execute' : ['execution', 'start', 'effective'],
    'expire' : ['expire', 'end', 'until'],
    'terminate' : ['terminate', 'cancel'],
    'renew' : ['renew', 'restart']
}

# cretae document segmentation model
def create_sentence_split_model():
    nlp = English()
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    return nlp

# create small english model
def create_sm_eng_model():
    return spacy.load("en_core_web_sm")

# create medium english model
def create_md_eng_model():
    return spacy.load("en_core_web_md")

# create large english model
def create_lg_eng_model():
    return spacy.load("en_core_web_lg")

# preprocessig of input plain text
def preproc_texts(plain_text):
    texts = plain_text.replace('\n', '. ').replace('\r', '').replace('  ', ' ').replace('\t', '. ')
    return texts

# convert words of date_class to their tokens
def convert_date_class_to_tokens(model, dclass):
    res = {}
    for c, words in dclass.items():
        l_tokens = []
        for word in words:
            tokens = model(word)
            l_tokens.append(tokens[0])
        res[c] = l_tokens
    return res

# split document into sentences
def split_plaintext_to_sents(sentence_model, texts):
    doc = sentence_model(texts)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences

# get doc of sentence
def get_doc_from_sentence(model, sentence):
    return model(sentence)

# extract dates from docs
# input : list[doc]
# output : list[dict]
def extract_dates_from_docs(docs):
    dates = []
    for idx, doc in enumerate(docs):
        for ent in doc.ents:
            if ent.label_ != "DATE":
                continue
            dates.append([idx, ent.text, ent.start_char, ent.end_char])
    return dates

# get date class of token
def get_dclass_from_token(date_tokens, inp_token):
    score_threshold = 0.7

    max_score = 0.0
    max_class = 'other'
    for c, tokens in date_tokens.items():
        similarities = [token.similarity(inp_token) for token in tokens]
        similarities = np.array(similarities)
        similarity = similarities.max()

        if max_score < similarity:
            max_score = similarity
            max_class = c

    if max_score < score_threshold:
        return 'other', max_score
    return max_class, max_score

# filter extracted dates like annual, monthly, ...
def filter_dates(dates):
    numbers = '0123456789'
    monthes = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    out = []

    for date in dates:
        txt = date[1]
        is_n = 0
        is_mon = False

        for s in txt:
            if s in numbers:
                is_n += 1

        for mon in monthes:
            if mon.lower() in txt.lower():
                is_mon = True
                break
        if is_n > 3 and is_mon:
            out.append(date)

    return out

# remove duplicated dates and add concepts from extracted dates
def post_proc_dates(sentences, dates):
    n = len(sentences)
    for date in dates:
        idx = date[0]
        if idx < n / 5 and len(sentences[idx]) < len(date[1]) * 2:
            date[4] = 'execute'
            date[5] = 2.0
        elif idx > n * 4 / 5 and len(sentences[idx]) < len(date[1]) * 2:
            date[4] = 'expire'
            date[5] = 2.0



# classify the date as excution, expire, terminate and renewal onr
# dclasses = dict<list>
def get_class_of_dates(docs, dates, date_tokens):
    for my_date in dates:
        idx = my_date[0]

        # scan the sentence of date
        max_c, max_score = 'other', 0
        for token in docs[idx]:
            """if token.is_stop:
                #print ('stop word : {}'.format(token.text))
                continue"""
            if (not token) or (not token.vector_norm):
                continue

            c, score = get_dclass_from_token(date_tokens, token)
            if max_score < score:
                max_score = score
                max_c = c
        my_date.append(max_c) # idx, text, start_char, end_char, date_class, date_class_score
        my_date.append(max_score)

def extract_dates(word_model, sentences):
    # get tokens of sentences
    docs = [get_doc_from_sentence(word_model, sent) for sent in sentences]

    # get date class tokens
    date_tokens = convert_date_class_to_tokens(word_model, date_classes)

    # extract dates from docs
    dates = extract_dates_from_docs(docs)
    dates = filter_dates(dates)

    # classification of extracted dates
    get_class_of_dates(docs, dates, date_tokens)

    post_proc_dates(sentences, dates)

    return dates


if __name__ == '__main__':
    f = open("1.txt", "r", encoding="utf8")
    plain_txt = f.read()
    plain_txt = preproc_texts(plain_txt)

    # load nlp models
    sentence_model = create_sentence_split_model()
    sm_eng_model = create_lg_eng_model()

    # split plain text to sentences
    sentences = split_plaintext_to_sents(sentence_model, plain_txt)

    dates = extract_dates(sm_eng_model, sentences)






    # show the result
    for date in dates:
        idx = date[0]
        print (sentences[idx])
        print (date[1], date[4], date[5])
        print ('-' * 10)


