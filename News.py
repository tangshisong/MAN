#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import json
from collections import Counter
from itertools import chain
from nltk.tokenize import word_tokenize
punc = [',','.',':',';','!','?','-','"','(',')','«','»','–']
categoty = {'nyheter':1,'bil':2,'abonnement':3,'meninger':4,'pluss':5,'vaeret':6,'student':7,
'100sport':8,'bedriftsannonser':9,'migration catalog':10,'tjenester':11,'bolig':12,'forbruker':13,
'streaming':14,'tema':15,'omadresseavisen':16,'sport':17,'kultur':18}
G = []

class News(object):
    def __init__(self,word2id=None,news=None,news2id=None,user_dict=None,behavior=None):
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<PAD>'] = 0
            self.word2id['<UNK>'] = 1
        self.id2word = {v: k for k,v in self.word2id.items()}
        if news:
            self.news = news
        else:
            self.news = dict() #idx->[[category],[title],[key]]
            self.news[0] = [[],[],[]]# 新闻必须从1计数，0作为保留位
        if news2id:
            self.new2sid = news2id
        else:
            self.news2id = dict()
            self.news2id['0'] = 0
        if user_dict:
            self.user_dict = user_dict
        else:
            self.user_dict = dict()
        if behavior:
            self.behavior = behavior
        else:
            self.behavior = dict()

    def add_word(self,word):
        if word not in self.word2id:
            id = self.word2id[word] = len(self.word2id)
            self.id2word[id] = word
            return id
        else:
            return self.word2id[word]

    def read_news_corpus(self,file_path):
        #eventid	time_x	userid	sessionstart_x	sessionstop_x	newsid	title	category	keyword
        news_corpus = []
        for line in open(file_path,encoding='utf-8'):
            title_token = key_token = category_token = []
            news_line = line.strip().split('\t')
            if news_line[2] not in self.user_dict:
                self.user_dict[news_line[2]] = len(self.user_dict)
            if news_line[5] not in self.news2id:
                self.news2id[news_line[5]] = len(self.news2id)

                title_token = [word.lower() for word in word_tokenize(news_line[6]) if word not in punc]
                if news_line[7] != 'null':
                    category_token = news_line[7].split('|')
                if news_line[8] != 'null':
                    key_token = [word.lower() for word in word_tokenize(news_line[8]) if word not in punc]
                assert key_token != ['null']

                self.news[len(self.news)] = [category_token,title_token,key_token]
                '''
                if len(title_token)==0:
                    print('error')
                    print(len(self.news))
                    print(news_line[6])
                    print(news_line[5])
                '''
                news_corpus.append(title_token)
                if len(key_token) > 0:
                    news_corpus.append(key_token)
        return news_corpus

    def build_behavior(self,file_path):
        behavior = {i:[] for i in range(len(self.user_dict))}
        for line in open(file_path,encoding='utf-8'):
            news_line = line.strip().split('\t')
            behavior[self.user_dict[news_line[2]]] += [self.news2id[news_line[5]]]

        #res = {val:key for key,val in self.user_dict.items()}
        for key,val in behavior.items():
            #x = len(behavior[key])
            behavior[key] = list(set(val))
            if len(behavior[key]) < 10:
                print("error")
                #G.append(res[key])

        self.behavior = behavior

    def build_vocab(self,news_corpus,vocab_size,cutoff = 2):
        word2freq = Counter(chain(*news_corpus))
        word = [k for k,v in word2freq.items() if v >= cutoff]

        print("Total vocabulary:{}. frequency >= {}:{}".format(
            len(word2freq),cutoff,len(word)))

        sort_word = sorted(word, key=lambda w: word2freq[w], reverse=True)[:vocab_size]
        for word in sort_word:
            self.add_word(word)

    def pad_sent(self,sents,pad_idx):
        sents_pad = []
        max_len = max([len(_) for _ in sents])
        for sent in sents:
            cur_len = len(sent)
            sents_pad.append([sent + [pad_idx]*(max_len-cur_len)])
        return sents_pad

    def words_to_idx(self,sents):
        if type(sents[0]) == list:
            return [[self.word2id[w] for w in s] for s in sents]
        else:
            return [self.word2id[w] if w in self.word2id else self.word2id['<UNK>']
                    for w in sents]

    def change(self):
        news_title = [[0]*10]
        news_key = [[0]*10]
        news_category = [[0]]
        for id,news in self.news.items():
            if id == 0:
                continue
            if len(news[0]) == 0:
                news_category.append([0])
            else:
                news_category.append([categoty[news[0][0]]])
            #print(news[1])
            t = self.words_to_idx(news[1])[:10]
            news_title.append(t+[0]*(10-len(t)))
            if len(news[2]) == 0:
                news_key.append([0]*10)
            else:
                t = self.words_to_idx(news[2])[:10]
                news_key.append(t+[0]*(10-len(t)))
        return news_title,news_key,news_category

    @classmethod
    def sents_to_tensor(self,sents_word,device):
        sents_idx = self.words_to_idx(sents_word)
        #sents_pad = self.pad_sent(sents_idx,self.word2id['<PAD>'])
        sents_tensor = torch.tensor(sents_idx,dtype=torch.long,device=device)
        return sents_tensor

    def save(self,file_path,news_title,news_key,news_category):
        json.dump(dict(word2id=self.word2id,
                       news=self.news,
                       news2id=self.news2id,
                       user_dict=self.user_dict,
                       behavior=self.behavior,
                       news_title=news_title,
                       news_key=news_key,
                       news_category=news_category),
                  open(file_path,'w',encoding='utf-8'),indent=2)

    @staticmethod
    def load(file_path):
        entry = json.load(open(file_path, 'r',encoding='utf-8'))
        return entry['word2id'],entry['news'],entry['news2id'],entry['user_dict'],\
               entry['behavior'],entry['news_title'],entry['news_key'],entry['news_category']


if __name__ == '__main__':
    news = News()
    news_corpus = news.read_news_corpus(file_path='newdata10.json')
    news.build_vocab(news_corpus,14000,1)
    news.build_behavior('newdata10.json')
    news_title,news_key,news_category=news.change()
    news.save('parameter.json',news_title,news_key,news_category)
    #json.dump(dict(G=G),open('G1.json','w',encoding='utf-8'),indent=2)