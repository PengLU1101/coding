# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 13:16:45 2016

@author: tjw
"""

import pandas as pd
import os

path='/home/tjw/data/summ/neuralsum/neuralsum/cnn/'

subpath='test/'

files=os.listdir(path+subpath)
contents=[open(path+subpath+_file).read() for _file in files]
splits=[_content.split('\n\n') for _content in contents]

def process_sentence(raw):
    sentences=raw.split('\n')
    parts=[s.split('\t\t\t') for s in sentences]
    return parts

def process_entity(raw):
    entities=raw.split('\n')
    pairs=[(_entity[:_entity.index(':')],_entity[_entity.index(':')+1:]) for _entity in entities if ':' in _entity]
    dic={k:v for (k,v) in pairs}
    return dic

urls=[_split[0] for _split in splits]
sentences=[process_sentence(_split[1]) for _split in splits]
highlights=[_split[2].split('\n') for _split in splits]
entities=[process_entity(_split[3]) for _split in splits]

assert len(urls)==len(sentences)==len(highlights)==len(entities)

df=pd.DataFrame(data={'sentence':sentences,'highlight':highlights,'entity':entities},index=urls)

df.to_pickle(path+subpath[:-1]+'.pkl')