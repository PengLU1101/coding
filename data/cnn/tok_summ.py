# -*- coding: utf-8 -*-
import sys
import hashlib
import glob
import os
import string


def Hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string.

    Args:
        s: The string to hash.

    Returns:
        A heximal formatted hash of the input string.
    """

    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


class ProgressBar(object):
    """Simple progress bar.

    Output example:
        100.00% [2152/2152]
    """

    def __init__(self, total=100, stream=sys.stderr):
        self.total = total
        self.stream = stream
        self.last_len = 0
        self.curr = 0

    def Increment(self):
        self.curr += 1
        self.PrintProgress(self.curr)

        if self.curr == self.total:
            print ''

    def PrintProgress(self, value):
        self.stream.write('\b' * self.last_len)
        pct = 100 * self.curr / float(self.total)
        out = '{:.2f}% [{}/{}]'.format(pct, value, self.total)
        self.last_len = len(out)
        self.stream.write(out)
        self.stream.flush()


def organize_summary(summ_dir):
    print 'Organizing summaries from: ', summ_dir
    file_list = glob.glob(summ_dir + "/*.proc")
    progress_bar = ProgressBar(len(file_list))
    docs = {}
    for summ in file_list:
        _, tail = os.path.split(summ)
        sid = tail.split('.')[0]
        try:
            newlines = []
            with open(summ, 'r') as f:
                lines = f.readlines()
                st = False
                for i, line in enumerate(lines):
                    if len(line.strip()) == 0:
                        if st is False:
                            st = True
                            newlines.append(line)
                    else:
                        st = False
                        newlines.append(line)
                del lines
            doc = {"url": "", "story": "", "tokens": [], "tok_summs": [], "summ_sents": []}
            cnt = 0
            for i, line in enumerate(newlines):
                if len(line.strip()) == 0:
                    cnt += 1
                else:
                    if cnt == 0:
                        doc["url"] = line
                    if cnt == 1:
                        doc["story"] = line
                    if cnt == 2:
                        doc["tokens"].append(line)
                    if cnt == 3:
                        doc["tok_summs"].append(line)
                    if cnt == 4:
                        doc["summ_sents"].append(line)
            docs[sid] = doc
        except IOError:
            print 'error: ', summ
        progress_bar.Increment()
    return docs


def tokenize_summary(proc_dir, docs):
    progress_bar = ProgressBar(len(docs))
    print 'Tokenizing summaries to', proc_dir
    for sid, doc in docs.items():
        with open(proc_dir + '/' + sid + '.fina', 'w+') as pf:
            pf.write(doc['url'] + '\n')
            pf.write(doc['story'].replace(" . ", " <eos> ").replace(" ? ", " <eos> ").replace(" ! ", " <eos> ").replace(" ; ", " <eos> ").replace(" .\n", " <eos>").replace(" ?\n", " <eos>").replace(" !\n", " <eos>").replace(" ;\n", " <eos>").replace("\n", " <eos>") + " <eod>\n\n")
            for token in doc["tokens"]:
                pf.write(token)
            pf.write('\n')
            summ_sents = []
            for ind, su in enumerate(doc["summ_sents"]):
                if len(su.strip()) == 0:
                    continue
                nsu = replaceDigits(su)
                nsu = removerep(nsu)
                nsu = clearentity(nsu)
                summ_sents.append(nsu)
                pf.write(nsu)
            pf.flush()
        progress_bar.Increment()


def replaceDigits(sent):
    xsent = sent.split(' ')
    for ind, word in enumerate(xsent):
        xword = list(word)
        if not word.strip().startswith('@'):
            for j, c in enumerate(word):
                if c.isdigit():
                    xword[j] = '#'
        xsent[ind] = ''.join(xword)
    return ' '.join(xsent)


def removerep(sent):
    xsent = list(sent)
    for i, c in enumerate(xsent):
        if c in string.punctuation and xsent[i + 1] in string.punctuation:
            del xsent[i + 1]
        if c is ' ' and xsent[i + 1] is ' ':
            del xsent[i + 1]
    return ''.join(xsent)


def clearentity(sent):
    xsent = sent.split(' ')
    for ind, word in enumerate(xsent):
        if word.startswith("@entity"):
            suffix = word[7:].strip()
            for i, c in enumerate(suffix):
                if not c.isdigit():
                    num = suffix[0:i]
                    xsent[ind] = "@entity" + num
                    break
    return ' '.join(xsent)


def test():
    summ_dir = "./data/processed/"
    fina_dir = "./data/final/"
    docs = organize_summary(summ_dir)
    tokenize_summary(fina_dir, docs)


def main(corp):
    base_dir = "/data/rali5/Tmp/pandu/summar/"
    stages = ["summaries", "processed", "final"]
    sets = ["training", "validation", "test"]
    path_from = base_dir + "/" + corp + "/" + stages[1] + "/"
    path_to = base_dir + "/" + corp + "/" + stages[2] + "/"
    from_dirs = [path_from + ent + "/" for ent in sets]
    to_dirs = [path_to + ent + "/" for ent in sets]

    for ind, summ_dir in enumerate(from_dirs):
        docs = organize_summary(summ_dir)
        tokenize_summary(to_dirs[ind], docs)


if __name__ == '__main__':
    # test()

    corp = 'cnn'
    # corp = 'dailymail'
    main(corp)
