#!/usr/bin/env python
# coding: utf-8

''' 
Author: Fabio Curi Paixao 
fcuri91@gmail.com
 '''

def normalize(dictionary, dic):

    normalized = defaultdict(list)
    missing = []

    for i, l in enumerate(dictionary.keys()):
        for e in dictionary[l]:
            tf, tg = [], []
            word1, word2 = e[0].replace('-', ' '), e[1].replace('-', ' ')

            if word1 in dic.keys():
                tf += [dic[word1]]
            if word1.lower() in dic.keys():
                tf += [dic[word1.lower()]]
            if word2 in dic.keys():
                tg += [dic[word2]]
            if word2.lower() in dic.keys():
                tg += [dic[word2.lower()]]

            if not tf:
                missing.append(word1.lower())

            if not tg:
                missing.append(word2.lower())

            if not tf:
                tf.append('-')
            if not tg:
                tg.append('-')

            for tf_ in list(set(tf)):
                for tg_ in list(set(tg)):
                    normalized[l].append((tf_, tg_))

    print(list(set(missing)))
        
    return normalized

if '__main__' == __name__:

    # Export positive sentences
    df = pd.read_csv('./test/merged_data.csv')
    export, final = [], []
    values = df.values.tolist()
    for l in values:
        if str(l[3]) != 'none':
            export.append(l[7] + '\t' + str(l[3]).upper() + '\t' + str(l[5]) + '\t' + str(l[6]) + '\t' + str(l[4]))
        
    export = list(set(export))
    final.append('#PMID:Sentence\tTagRNN\tTF\tTG\tSentence')
    final += export
    write_list(final, './test/predictions.txt', True, 'latin-1')

    silver_standard = read_as_list('ExTRI_confidence', encoding='latin-1')[2:]

    d_silver, d_predicted = defaultdict(list), defaultdict(list)

    for l in silver_standard:
        l_ = l.split('\t')
        if float(l_[3])>float(0.7):
            d_silver[':'.join(l_[0].split(':')[0:2])].append((l_[1], l_[2]))

    for l in final:
        l_ = l.split('\t')
        d_predicted[l_[0]].append((l_[2], l_[3]))

    dic = eval(open("../dictionaries/all_dics.txt2", "r").read())

    n_silver = normalize(d_silver, dic)
    n_predicted = normalize(d_predicted, dic)

    # Label as TP and FN
    labels = []
    for key in n_silver.keys():
        for a in n_silver[key]:
            if a in n_predicted[key]:
                labels.append('TP')
            else:
                labels.append('FN')

    c = Counter(labels)
    FN = c['FN']

    # Label as TP and FP
    labels = []
    for key in n_predicted.keys():
        for a in n_predicted[key]:
            if a in n_silver[key]:
                labels.append('TP')
            else:
                labels.append('FP')

    c = Counter(labels)
    FP = c['FP']
    TP = c['TP']
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    f1 = 2*precision*recall/(precision+recall)
    print('recall ' + str(recall))
    print('precision ' + str(precision))
    print('f1-score ' + str(f1))
