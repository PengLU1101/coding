# -*- coding: utf-8 -*-
import sys
import hashlib
import glob
import os


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


def mergeQuestion2Summary(corp_dir, summs, merg_dir):
    print 'Reading questions from: ', corp_dir
    file_list = glob.glob(corp_dir + "/*.question")
    progress_bar = ProgressBar(len(file_list))
    qas = {}
    conts = {}
    for que in file_list:
        try:
            with open(que, 'r') as f:
                lines = f.readlines()
                sid = Hashhex(lines[0].strip())
                if sid in qas:
                    qas[sid].append((lines[4], lines[6]))
                else:
                    qas[sid] = [(lines[4], lines[6])]
                del lines[6]
                del lines[4]
                if sid not in conts:
                    conts[sid] = lines
        except IOError:
            print 'error: ', que
        progress_bar.Increment()
    print 'Merging question and highlights to summaries'
    progress_bar2 = ProgressBar(len(conts))
    for key, cont in conts.items():
        alllines = cont
        if key in qas:
            alllines.append('\n')
            for q, a in qas[key]:
                alllines.extend([q, a])
            if key in summs:
                alllines.append('\n')
                alllines.extend(summs[key])
        with open(merg_dir + "/" + key + '.summ', 'w+') as sumf:
            for line in alllines:
                if not line.endswith('\n'):
                    line += '\n'
                sumf.write(line)
            sumf.flush()
        progress_bar2.Increment()


def getsummary_map(story_dir):
    print 'Getting summary map from: ', story_dir
    file_list = glob.glob(story_dir + "/*.story")
    progress_bar = ProgressBar(len(file_list))
    summary_map = {}
    for story in file_list:
        _, tail = os.path.split(story)
        sid = tail.split('.')[0]
        try:
            with open(story, 'r') as f:
                lines = f.readlines()
                summary = []
                for i, line in enumerate(lines):
                    if line.startswith("@highlight"):
                        summary.append(lines[i + 2])
                summary_map[sid] = summary
        except IOError:
            print 'error: ', story
        progress_bar.Increment()
    return summary_map

if __name__ == '__main__':
    # corp = 'cnn'
    corp = 'cnn'
    story_dir = './data/rali5/Tmp/pandu/summar/' + corp + '/stories/'

    corp_dirs = ['/data/rali5/Tmp/pandu/summar/' + corp + '/questions/training/', '/data/rali5/Tmp/pandu/summar/' + corp + '/questions/test/', '/data/rali5/Tmp/pandu/summar/' + corp + '/questions/validation/']
    merg_dirs = ['/data/rali5/Tmp/pandu/summar/' + corp + '/summaries/training/', '/data/rali5/Tmp/pandu/summar/' + corp + '/summaries/test/', '/data/rali5/Tmp/pandu/summar/' + corp + '/summaries/validation/']

    summs = getsummary_map(story_dir)
    for ind, corp_dir in enumerate(corp_dirs):
        print 'processing', corp_dir
        mergeQuestion2Summary(corp_dir, summs, merg_dirs[ind])
