import spacy
import pickle
import numpy as np
import warnings
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from cname import preproc_texts, split_plaintext_to_sents
from mdate import create_sentence_split_model

# new entity label
LABEL = "MADDRESS"

# read train dataset
def load_train_dataset(pkl_file):
    dataset_file = open(pkl_file, 'rb')
    dataset_list = pickle.load(dataset_file)
    return dataset_list

# convert pkl file to spacy ner dataset format
def convert_pkl_spacy_format(dataset_list):
    dataset_spacy = []
    for rec in dataset_list:
        sent = ''
        start_char = []
        end_char = []
        idx = 0
        for part1 in rec:
            word = part1[0][0]
            cls = part1[1]

            if word[0] != ',' and word[0] != '.':
                word = ' ' + word
            sent = sent + word
            if cls == 'B-GPE':
                if len(start_char) > len(end_char):
                    end_char.append(idx)
                start_char.append(idx)
            if len(start_char) > len(end_char) and cls == 'O':
                end_char.append(idx-1)

            idx += len(word)

        if len(start_char) > len(end_char):
            end_char.append(idx)

        ents = []
        for i in range(len(start_char)):
            ents.append((start_char[i], end_char[i], LABEL))

        sent = sent[1:]
        dataset_spacy.append((sent, {"entities" : ents}))

    return dataset_spacy

# check pkl dataset
def check_pkl_dataset(pkl_file):
    dataset_list = load_train_dataset(pkl_file)
    kinds = []
    max_ents = 0
    max_ent_sent = ''
    n_multiple_ents = 0

    dataset_len = len(dataset_list)

    for rec in dataset_list:
        n_ents = 0
        for part in rec:
            lbl = part[1]
            if not lbl in kinds:
                kinds.append(lbl)
            if lbl == 'B-GPE':
                n_ents += 1
        if n_ents > max_ents:
            max_ents = n_ents
            max_ent_sent = rec
        if n_ents > 1:
            n_multiple_ents += 1

    print ('length : {}'.format(dataset_len))
    print ('ents : {}'.format(kinds))
    print ('max ent num : {}'.format(max_ents))
    print ('max ent sentence : {}'.format(max_ent_sent))
    print ('number of multiple ents : {}'.format(n_multiple_ents))

# check spacy dataset
def check_spacy_dataset(dataset):
    for rec in dataset:
        txt = rec[0]
        for ent in rec[1]['entities']:
            s = ent[0]
            e = ent[1]
            print (txt[s:e])

# cut out first numbers
def cut_first_numbers(tstr):
    nums = '0123456789, '
    idx =0
    while True:
        if not tstr[idx] in nums:
            break
        idx += 1
    return idx, tstr[idx:]


# augment spacy dataset
# cut out the first numbers of the address
def augment_spacy_dataset(spacy_dataset):
    augmented = []

    for rec in spacy_dataset:
        sent = rec[0]
        ents = rec[1]['entities']
        aug_ents = []
        ncut = 0
        for ent in ents:
            s = ent[0] - ncut
            e = ent[1] - ncut
            addr = sent[s:e]
            l_num, c_addr = cut_first_numbers(addr)
            e -= l_num
            sent = sent[:s] + sent[s+l_num:]
            ncut += l_num
            aug_ents.append((s, e, LABEL))
        if ncut > 0:
            augmented.append((sent, {'entities' : aug_ents}))
    return augmented



# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    (
        "Horses are too tall and they pretend to care about your feelings",
        {"entities": [(0, 6, LABEL)]},
    ),
    ("Do they bite?", {"entities": []}),
    (
        "horses are too tall and they pretend to care about your feelings",
        {"entities": [(0, 6, LABEL)]},
    ),
    ("horses pretend to care about your feelings", {"entities": [(0, 6, LABEL)]}),
    (
        "they pretend to care about your feelings, those horses",
        {"entities": [(48, 54, LABEL)]},
    ),
    ("horses?", {"entities": [(0, 6, LABEL)]}),
]


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
    ner.add_label("VEGETABLE")
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

        sizes = compounding(1.0, 300.0, 1.001)
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
    test_text = "This AGREEMENT is by and between A.cme Company1, LLC.(A.cme), a Delaware limited liability company having a place of business at 5523 Technology Drive, Lowell, MA 11111 and A.cme Company 2 Services, Inc.(Customer), having a place of business at 1593 Spring Hill Rd #600, Vienna, VA 22182"
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

# load model
def load_model(model_path):
    model = spacy.load(model_path)
    return model



# extract first 20 lines
def extract_first_lines(text):
    arr = text.split('\n')
    res = ''
    for each in arr[:20]:
        res = res + each + '\n'
    return res

# filter the addresses
def filter_addresses(addrs):
    f_addrs = []

    for addr in addrs:
        txt = addr[1]
        # remove if addr contain .
        if '.' in txt:
            continue
        # remove if addr doesn't contain ,
        if not ',' in txt:
            continue

        is_word = False
        is_f_num = False
        arr1 = txt.split(' ')
        for each in arr1:
            arr2 = each.split(',')
            for each2 in arr2:
                if each2.isalpha():
                    is_word = True
                if each2.isnumeric() and len(each2) == 5:
                    is_f_num = True
                if is_word and is_f_num:
                    break
        if (not is_word) or (not is_f_num):
            continue
        f_addrs.append(addr)
    return f_addrs

# extract address from sentences
def extract_address(addr_model, sentences):
    res = []
    for idx, sent in enumerate(sentences):
        doc = addr_model(sent)
        for ent in doc.ents:
            if ent.label_ == LABEL:
                res.append([idx, ent.text, ent.start_char, ent.end_char])
    res = filter_addresses(res)
    return res

if __name__ == '__main__':
    if False:
        pkl_file = "IOB_tagged_addresses.pkl"
        dataset_list = load_train_dataset(pkl_file)
        dataset_spacy = convert_pkl_spacy_format(dataset_list)
        #check_spacy_dataset(dataset_spacy)
        #check_pkl_dataset(pkl_file)
        augmented_dataset = augment_spacy_dataset(dataset_spacy)
        #print (len(augmented_dataset))
        #print (augmented_dataset[13])
        #exit()
        dataset_spacy = dataset_spacy + augmented_dataset
        print (len(dataset_spacy))
        train(dataset_spacy, output_dir='address_model_aug_30', n_iter=15, model='address_model_aug_15')
    else:
        model = create_sentence_split_model()
        f = open("3.txt", 'r',  encoding='utf8')
        p_text = f.read()
        f.close()
        p_text = preproc_texts(p_text)
        addr_model = load_model('address_model_aug_30')

        #sents = [p_text]
        sents = split_plaintext_to_sents(model, p_text)

        addrs = extract_address(sents, addr_model)
        print (addrs)



