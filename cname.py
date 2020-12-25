from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import random
import pandas as pd
import warnings
import math

#LABEL = "COMPANYNAME"
LABEL = 'ORG'

CompanyInc = ['inc', 'llc', 'corporation', 'corp', 'company', 'co', 'ltd', 'lp', 'bank', 'group']
# convert spacy dataset from csv
def convert_csv_to_spacy_dataset(dataframe):
    dataset_spacy = []
    for idx, row in dataframe.iterrows():
        sent = row[0].strip()
        tags = row[1]
        if not isinstance(tags, str):
            dataset_spacy.append((sent, {'entities' : []}))
            continue

        tags = tags.strip().replace('   ', ' ').replace('  ', ' ')

        arr = tags.split(' ')
        if (len(arr) % 2) == 1:
            print (idx)
            print (tags)
            return None

        start_ids = arr[::2]
        end_ids = arr[1::2]
        ents = []
        for i in range(len(start_ids)):
            ents.append((int(start_ids[i]), int(end_ids[i]), LABEL))

        dataset_spacy.append((sent, {'entities' :ents}))
    return dataset_spacy

# filter the spacy dataset
# remove records that has ents of [0, x]
def filter_spacy_dataset(spacy_dataset):
    filtered_dataset = []
    for d in spacy_dataset:
        ents = d[1]['entities']
        if len(ents) == 0:
            continue
        if len(ents) == 1 and ents[0][0] == 0:
            continue
        filtered_dataset.append(d)
    return filtered_dataset

def train(dataset_spacy, model=None, new_model_name="animal", output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    ner.add_label("NONCOMPANYNAME")
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        sizes = compounding(1.0, 100.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(dataset_spacy)
            batches = minibatch(dataset_spacy, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.5, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = 'THIS AGREEMENT is by and between ACmeCompany1, LLC.("Contract Logix"), a Delaware limited liability company having a place of business at 248 Mill Road, Building 1, Unit 3, Chelmsford MA 01824, USA, and ACmeCompany2 Company, Inc. ("Licensee"), having a place of business at Ambest RD, Oldwick, NJ 08858.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)

# -----------------------------------------------------------------------------------------------------------------------------

# load model
def load_model(model_path):
    model = spacy.load(model_path)
    return model

# list of token of sentences
def get_list_tokens(model, sentence):
    doc = model(sentence)
    res = []
    for token in doc:
        res.append([token.text, token.idx, token.pos_, token.shape_, token.is_stop])
    return res

# get index of token from sentence position
def get_token_idx(token_list, pos):
    tid = -1
    for tid, token in enumerate(token_list):
        s = token[1]
        e = s + len(token[0])
        if s<=pos and pos < e:
            break
    return tid

# check if the token is attachable to the company name
def check_attach(ltoken):
    word = ltoken[0]
    pos = ltoken[2]
    shape = ltoken[3]
    bstop = ltoken[4]

    # stop word can't be attached
    if bstop:
        return False
    if pos in ['DET', 'ADP', 'CCONJ', 'VERB', 'AUX']:
        return False
    # if lower case letter, can't be attached
    if shape[0] == 'x':
        return False
    # only & punctuation can be attached
    if pos == 'PUNCT' and word != '&':
        return False
    return True


# extract company names from sentences
def extract_company_names(sentences, cname_model):
    res = []
    for idx, sent in enumerate(sentences):
        doc = cname_model(sent)
        for ent in doc.ents:
            if ent.label_ == LABEL:
                res.append([idx, ent.text, ent.start_char, ent.end_char])
    return res

# expand chunk of company names
# assume : company_name is ent type
def expand_chunk_company_names(sentences, company_names, model):
    expanded = []
    for cname in company_names:
        idx = cname[0]
        n = cname[1]
        s = cname[2]
        e = cname[3]

        sent = sentences[idx]
        token_list = get_list_tokens(model, sent)

        back_id = -1
        forward_id = -1
        # backward propagation
        wid = get_token_idx(token_list, s)
        i = -1
        for i in range(wid-1, -1, -1):
            ltoken = token_list[i]
            if ltoken[0] in [',', '.']:
                if i == 0:
                    break # can't attach
                if check_attach(token_list[i-1]) and i < len(token_list) - 1 and (token_list[i+1][0].lower() in ['com', 'com.', 'inc', 'inc.', 'ltd', 'ltd.', 'co', 'co.', 'llc', 'llc.']):
                    continue # attach
            if not check_attach(ltoken):
                break
        back_id = i + 1

        # forward propagation
        wid = get_token_idx(token_list, e-1)
        i = len(token_list)
        for i in range(wid + 1, len(token_list), 1):
            ltoken = token_list[i]
            if ltoken[0] in [',', '.']:
                if i == len(token_list) - 1:
                    break
                if token_list[i+1][0].lower() in ['com', 'com.', 'inc', 'inc.', 'ltd', 'ltd.', 'co', 'co.', 'llc', 'llc.']:
                    continue
            if not check_attach(ltoken):
                break
        forward_id = i - 1

        s = token_list[back_id][1]
        e = token_list[forward_id][1] + len(token_list[forward_id][0])
        n = sent[s:e]
        expanded.append([idx, n, s, e])

    return expanded

# check if start after soom space
def check_start_soon(each):
    if len(each) < 4:
        return True
    if each[0] == '\t' or each[0] == '\r':
        return False
    if each[0] == ' ' and each[1] == ' ' and each[2] == ' ':
        return False
    return True

# preprocessig of input plain text
def preproc_texts(plain_text):
    # insert . in lines that start from start_pos and has length than 100
    len_thres = 100
    arr = plain_text.split('\n')
    n_arr = []
    for i in range(len(arr)):
        each = arr[i]
        l = len(each.strip())
        if l > len_thres:
            n_arr.append(each)
            continue
        if not check_start_soon(each):
            if i < len(arr) - 1 and check_start_soon(arr[i+1]):
                n_arr.append(each + ' .')
                continue
        else:
            n_arr.append(each + ' .')
            continue
        n_arr.append(each)

    plain_text = ''
    for each in n_arr:
        plain_text += each + '\n'

    # insert . in lines that has no letters
    arr = plain_text.split('\n')
    n_arr = []
    for each in arr:
        if each.strip() == '':
            n_arr.append(' .')
        else:
            n_arr.append(each)
    plain_text = ''
    for each in n_arr:
        plain_text += each + '\n'
    # remove ()
    s_brackets = []
    e_brackets = []
    for pos, c in enumerate(plain_text):
        if c == '(' or c == '[' or c == '{' or c == '<': s_brackets.append(pos)
        elif c == ')' or c == ']' or c == '}' or c == '>': e_brackets.append(pos)

    if len(s_brackets) != len(e_brackets):
        print ('bracket number error')
    else:
        for i in range(len(s_brackets) - 1, -1, -1):
            plain_text = plain_text[:s_brackets[i]] + plain_text[e_brackets[i]+1:]


    texts = plain_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('  ', ' ')
    texts = texts.replace('  ', ' ')
    return texts

# split document into sentences
def split_plaintext_to_sents(sentence_model, texts):
    doc = sentence_model(texts)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences

# filter company names
def filter_company_names(cnames):
    filtered = []
    for cname in cnames:
        word = cname[1]
        w = word.replace('0', ' ').replace('1', ' ').replace('2', ' ').replace('3', ' ').replace('4', ' ').replace('5', ' ').replace('6', ' ').replace('7', ' ').replace('8', ' ').replace('9', ' ')
        w = w.replace(',', ' ').replace('.', ' ').replace('  ', ' ')
        arr = w.split(' ')

        flg = False
        for each in arr:
            if each.lower() in CompanyInc:
                flg = True
                break
        if flg:
            if cname[1][-1] == '.':
                cname[1] = cname[1][:-1]
            filtered.append(cname)

    filtered2 = []
    for i in range(len(filtered)):
        w = filtered[i][1]
        flg = False
        for j in range(i):
            w2 = filtered[j][1]
            if w2 == w:
                flg = True
                break
        if not flg:
            filtered2.append(filtered[i])

    return filtered2

def extract_parties(word_model, sentences):
    company_names = extract_company_names(sentences, word_model)
    #print(company_names)
    #print('-' * 10)

    company_names = expand_chunk_company_names(sentences, company_names, word_model)
    company_names = filter_company_names(company_names)

    return company_names

if __name__ == '__main__':

    p_text = 'This Addendum to the License Agreement {"Addendum") is effective January 28, 2014 ("Effective Date") and is made by Acme Company1, Inc. ("Customer") and Acme Company2, LLC. ("Company2").'
    p_text2 = 'Customer and Company2 entered into that certain License Agreement effective September 05, 208 (Lhe "Original Agreement")'
    p_text3 = 'THIS AGREEMENT is by and between ACmeCompany1, LLC. ("Contract Logix"), a Delaware limited liability company having a place of business at 248 Mill Road, Building 1, Unit 3, Chelmsford MA 01824, USA, and ACmeCompany2 Company, Inc. ("Licensee"), having a place of business at Ambest RD, Oldwick, NJ 08858!.'
    p_text4 = '7301 Parkway Drive Hanover, MD 21076'

    if True:
        f = open("1.txt", "r", encoding="utf8")
        plain_txt = f.read()
        plain_txt = preproc_texts(plain_txt)
        print(plain_txt)
        #plain_txt = p_text4

        model = load_model('en_core_web_lg')

        sentences = split_plaintext_to_sents(model, plain_txt)
        #sentences = [p_text3]
        print (sentences[0])
        print (sentences[1])
        print (sentences[2])
        print (sentences[3])
        print (sentences[4])
        print (sentences[5])

        company_names = extract_parties(model, sentences)
        print (company_names)
        print ('-' * 10)

        """
        doc = model(sentences[1])
        for token in doc:
            print (token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, token.i, token.idx)"""
        exit()

    if True:
        csv_path = 'company_name.csv'
        df_csv = pd.read_csv('company_name.csv', encoding='cp1252')
        print (df_csv.head())

        dataset_spacy = convert_csv_to_spacy_dataset(df_csv)
        print (len(dataset_spacy))
        filtered_dataset = filter_spacy_dataset(dataset_spacy)
        print (len(filtered_dataset))
        print (filtered_dataset[0])

        train(dataset_spacy, output_dir='cname_model_aug_30', n_iter=30)
    else:
        cname_model = load_model('cname_model_aug_30')
        p_text = 'This agreement is by and between ACmeCompany1, LLC.("Contract Logix"), a Delaware limited liability company having a place of business at 248 Mill Road, Building 1, Unit 3, Chelmsford MA 01824, USA, and ACmeCompany2 Company, Inc. ("Licensee"), having a place of business at Ambest RD, Oldwick, NJ 08858.'
        cnames = extract_company_names([p_text], cname_model)
        print (cnames)




