import argparse
from mdate import create_sentence_split_model, create_lg_eng_model, split_plaintext_to_sents, extract_dates
from maddress import load_model, extract_address
from cname import preproc_texts, extract_parties
from mdoc import get_class


# load models for extraction
def load_models():
    #sent_split_model = create_sentence_split_model()
    sent_split_model = load_model('en_core_web_md')
    word_model = create_lg_eng_model()
    address_model = load_model('address_model_aug_30')

    return sent_split_model, word_model, address_model

def main(txt_path, cls_path, sent_split_model, word_model, address_model):
    f = open(txt_path, "r", encoding="utf8")
    plain_txt = f.read()
    plain_txt = preproc_texts(plain_txt)
    f.close()
    sentences = split_plaintext_to_sents(sent_split_model, plain_txt)

    #print (sentences)

    dates = extract_dates(word_model, sentences)
    addresses = extract_address(address_model, sentences)
    companies = extract_parties(word_model, sentences)
    doc_cls = get_class(word_model, sentences, cls_path)

    return dates, addresses, companies, doc_cls



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract company names, addresses, dates from text')
    parser.add_argument('-i', '--text_path', metavar='TEXT_PATH', type=str, default='2.txt', help='location of input text')
    parser.add_argument('-c', '--class_path', metavar='CLS_PATH', type=str, default='myclass.txt', help='location of document classes')
    args = parser.parse_args()

    sent_split_model, word_model, address_model = load_models()
    dates, addresses, companies, doc_cls =  main(args.text_path, args.class_path, sent_split_model, word_model, address_model)

    print ('-------companies-------')
    print (companies)

    print('-------addresses-------')
    print(addresses)

    print('-------dates-------')
    print(dates)

    print('-------doc class-------')
    print(doc_cls)


