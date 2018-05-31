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
    file_list = glob.glob(summ_dir + "/*.summ")
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
    ct = 0
    print 'Tokenizing summaries to', proc_dir
    for sid, doc in docs.items():
        with open(proc_dir + '/' + sid + '.proc', 'w+') as pf:
            pf.write(doc['url'] + '\n')
            pf.write(doc['story'] + '\n')
            tokens = {}
            for token in doc["tokens"]:
                pf.write(token)
                val, key = token.strip().split(":", 1)
                tokens[key.strip()] = val.strip()
            pf.write('\n')
            tok_summs = []
            for ind, ts in enumerate(doc["tok_summs"]):
                if len(ts.strip()) == 0:
                    continue
                if "@placeholder" in ts:
                    nts = ts.replace("@placeholder", doc["tok_summs"][ind + 1].strip())
                    tok_summs.append(nts)
                    pf.write(nts)
            pf.write('\n')
            summ_sents = []
            for ind, su in enumerate(doc["summ_sents"]):
                if len(su.strip()) == 0:
                    continue
                # nsu = splitpuncts(su)
                nsu = splitS(su)
                for entity in tokens.keys():
                    if entity in nsu:
                        nsu = nsu.replace(entity, tokens[entity])
                nsu = splitpuncts_lim(nsu)
                nsu = nsu.lower()
                summ_sents.append(nsu)
                pf.write(nsu)
            pf.flush()

            tok_s = [e.strip() for e in tok_summs if len(e.strip()) > 0]
            summ_s = [e.strip() for e in summ_sents if len(e.strip()) > 0]
            if not set(tok_s).issubset(set(summ_s)):
                ct += 1
                # print sid, len(set(tok_summs)), len(set(summ_s))
        progress_bar.Increment()
    print 'warning', ct, len(docs)


def lcs(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)
    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in xrange(m + 1)]
    """Following steps build L[m+1][n+1] in bottom up fashion
    Note: L[i][j] contains length of LCS of X[0..i-1]
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])
    # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]


def splitS(sent):
    xsent = sent.split(' ')
    for ind, word in enumerate(xsent):
        if word.strip().endswith('\'d') or word.strip().endswith('\'s') or word.strip().endswith('\'ll') or word.strip().endswith('\'re') or word.strip().endswith('\'ve'):
            pos = word.rfind('\'')
            a = word[0:pos] + ' ' + word[pos:]
            xsent[ind] = a
    return ' '.join(xsent)


def splitpuncts_lim(sent):
    lim = ['"', ',', ':']
    nsent = []
    sent = sent.strip()
    for i, c in enumerate(sent):
        if c in lim:
            if i == 0:
                nsent.append(c)
                nsent.append(' ')
            elif i == len(sent) - 1:
                nsent.append(' ')
                nsent.append(c)
            else:
                if sent[i - 1] is not ' ':
                    nsent.append(' ')
                    nsent.append(c)

                if sent[i + 1] is not ' ':
                    nsent.append(c)
                    nsent.append(' ')
        elif c == '\'':
            seg = sent[i:]
            if seg.startswith('\'d ') or seg.startswith('\'d\n') or seg.startswith('\'s ') or seg.startswith('\'s\n') or seg.startswith('\'ll ') or seg.startswith('\'ll\n') or seg.startswith('\'re ') or seg.startswith('\'re\n') or seg.startswith('\'ve ') or seg.startswith('\'ve\n'):
                nsent.append(c)
            else:
                if i == 0:
                    nsent.append(c)
                    nsent.append(' ')
                elif i == len(sent) - 1:
                    nsent.append(' ')
                    nsent.append(c)
                else:
                    if sent[i - 1] is not ' ':
                        nsent.append(' ')
                        nsent.append(c)

                    if sent[i + 1] is not ' ':
                        nsent.append(c)
                        nsent.append(' ')
        else:
            nsent.append(c)
    nsent.append('\n')
    return ''.join(c for c in nsent)


def splitpuncts(sent):
    nsent = []
    sent = sent.strip()
    for i, c in enumerate(sent):
        if c in string.punctuation and c is not '@':
            if i == 0:
                nsent.append(c)
                nsent.append(' ')
            elif i == len(sent) - 1:
                nsent.append(' ')
                nsent.append(c)
            else:
                if sent[i - 1] is not ' ':
                    nsent.append(' ')
                    nsent.append(c)

                if sent[i + 1] is not ' ':
                    nsent.append(c)
                    nsent.append(' ')

        else:
            nsent.append(c)
    nsent.append('\n')
    for i, c in enumerate(nsent):
        if c == '\'':
            if i < len(nsent) - 1:
                if nsent[i + 1] == '\'':
                    del nsent[i]
            if i < len(nsent) - 3:
                if nsent[i + 1] == ' ' and (nsent[i + 2] == 's' or nsent[i + 2] == 'd') and (nsent[i + 3] == ' ' or nsent[i + 3] == '\n'):
                    del nsent[i + 1]
            if i < len(nsent) - 4:
                if nsent[i + 1] == ' ' and (nsent[i + 2] == 'l' and nsent[i + 3] == 'l' or nsent[i + 2] == 'r' and nsent[i + 3] == 'e' or nsent[i + 2] == 'v' and nsent[i + 3] == 'e') and (nsent[i + 4] == ' ' or nsent[i + 4] == '\n'):
                    del nsent[i + 1]
        if c == '\"':
            if i < len(nsent) - 1:
                if nsent[i + 1] == '\"':
                    del nsent[i]
        if c == ',':
            if i < len(nsent) - 1:
                if nsent[i + 1] == ',':
                    del nsent[i]
        if c == '.':
            if i < len(nsent) - 1:
                if nsent[i + 1] == '.':
                    del nsent[i]
        if c == ':':
            if i < len(nsent) - 1:
                if nsent[i + 1] == ':':
                    del nsent[i]
    return ''.join(c for c in nsent)

if __name__ == '__main__':
    # summ_dir = "./data/summaries/"
    # proc_dir = "./data/processed/"
    corp = 'cnn'
    # corp = 'dailymail'
    sum_dirs = ['/data/rali5/Tmp/pandu/summar/' + corp + '/summaries/training/', '/data/rali5/Tmp/pandu/summar/' + corp + '/summaries/test/', '/data/rali5/Tmp/pandu/summar/' + corp + '/summaries/validation/']
    proc_dirs = ['/data/rali5/Tmp/pandu/summar/' + corp + '/processed/training/', '/data/rali5/Tmp/pandu/summar/' + corp + '/processed/test/', '/data/rali5/Tmp/pandu/summar/' + corp + '/processed/validation/']
    for ind, summ_dir in enumerate(sum_dirs):
        docs = organize_summary(summ_dir)
        tokenize_summary(proc_dirs[ind], docs)
