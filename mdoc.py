import spacy
from mdate import create_sentence_split_model
from cname import preproc_texts, split_plaintext_to_sents
import numpy as  np

# get list of tokens
def get_list_tokens(token_model, cls_path):
    dict_tokens = {}

    with open (cls_path, 'r', encoding='utf8') as f:
        while True:
            cls = f.readline()
            if not cls:
                break

            cls = cls.strip()
            if cls == '':
                continue
            tokens = token_model(cls)
            dict_tokens[cls] = tokens
    return dict_tokens

# get similiarith between token lists
def similarity_tokens(cls_tokens, doc_tokens, similarity_thres = 0.7, penalty_score = -1.0):
    #added = max(0, 1.0 - similarity_thres)
    added = 0.4
    score = 0
    nword = 0
    npositive = 0
    for cls_t in cls_tokens:
        if (not cls_t) or (not cls_t.vector_norm):
            continue
        sims = [cls_t.similarity(doc_t) for doc_t in doc_tokens]
        sims = np.array(sims)
        word_score = sims.max()

        if word_score > similarity_thres:
            score += word_score
            npositive += 1
        else:
            score -= penalty_score
        nword += 1

    if nword == 0:
        #print (cls_tokens)
        return 0

    return (score +  added * (npositive-1)) / nword

# get classes of first sentences
def get_class(token_model, sents, cls_path, nres = 5):
    dict_tokens = get_list_tokens(token_model, cls_path)

    len_thres = 50
    if len(sents[0]) < len_thres and len(sents[1]) < len_thres:
        nsents = 2
    else:
        nsents = 1

    doc_words = ''
    for sent in sents[:nsents]:
        doc_words = doc_words + ' ' + sent

    doc_tokens = token_model(doc_words)
    f_doc_tokens = []
    for doc_t in doc_tokens:
        if (not doc_t) or (not doc_t.vector_norm):
            continue
        f_doc_tokens.append(doc_t)


    # get doc class
    cls_scors = {}
    for c, ts in dict_tokens.items():
        score = similarity_tokens(ts, f_doc_tokens)
        cls_scors[c] = score

    #print (cls_scors)

    # sort result
    cls_scors = [k for k, v in sorted(cls_scors.items(), key=lambda cls:cls[1], reverse=True)]

    return cls_scors[:nres]

if __name__ == '__main__':
    f = open('1.txt', 'r', encoding='utf8')
    p_text = f.read()
    p_text = preproc_texts(p_text)
    f.close()

    sent_seg_model = create_sentence_split_model()
    word_sim_model = spacy.load('en_core_web_lg')
    sents = split_plaintext_to_sents(sent_seg_model, p_text)

    res_class = get_class(word_sim_model, sents, 'myclass.txt')
    print ('length of sentences : {}'.format(len(sents)))
    print (res_class)


