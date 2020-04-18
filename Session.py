#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import json
import random

class Session(object):
    def __init__(self,user_dict=None):
        super(Session, self).__init__()
        if user_dict:
            self.user_dict = user_dict
        else:
            self.user_dict = dict()

    def news_sample(self,data,ratio):
    # sampling news randomly
        if len(data) < ratio:
            return np.random.choice(data*(ratio//len(data)+1),ratio,replace=False).tolist()
        else:
            return np.random.choice(data,ratio,replace=False).tolist()


    def generate_data(self,behavior,news2id,nratio):
        train_news_id = []
        train_news_label = []
        train_user_id = []

        test_news_id = []
        test_news_label = []
        test_user_id = []
        test_pos_loc = []

        user_pos = []
        test_user_pos = []
        for user,news_pos in behavior.items():
            user = int(user)
            news_neg = [i for i in range(1,len(news2id)) if i not in news_pos ]
            assert len(news_neg) + len(news_pos) == len(news2id)-1

            news_neg = self.news_sample(news_neg,len(news_pos)*5)

            train_pos = self.news_sample(news_pos,len(news_pos)*3//5)
            test_pos = self.news_sample(list(set(news_pos).difference(train_pos)),len(news_pos)*2//5)

            train_neg = self.news_sample(news_neg,len(news_neg)*3//5)
            test_neg = self.news_sample(list(set(news_neg).difference(train_neg)),len(news_neg)*2//5)

            idx = sorted(random.sample([i for i in range(1,len(test_neg))],len(test_pos)-1))
            idx.append(len(test_neg))
            temp = []
            start = 0
            for i in idx:
                temp.append(test_neg[start:i])
                start = i

            i = 0
            for pos_sample in test_pos:
                session_idx = []
                session_idx.append(len(test_news_id))
                pos_set = list(set(train_pos))
                all_pos = [p for p in self.news_sample(pos_set, min(50, len(pos_set)))[:50]]
                all_pos += [0] * (50 - len(all_pos))

                test_news_id.append(int(pos_sample))
                test_news_label.append(1)
                test_user_id.append(user)
                test_user_pos.append(all_pos)

                assert temp[i] is not None

                for j in temp[i]:
                    test_news_id.append(int(j))
                    test_news_label.append(0)
                    test_user_id.append(user)
                    test_user_pos.append(all_pos)
                i += 1
                session_idx.append(len(test_news_id))
                test_pos_loc.append(session_idx)

            for pos_sample in train_pos:
                neg_sample = self.news_sample(train_neg,nratio)
                train_neg = list(set(train_neg)-set(neg_sample))
                neg_sample.append(pos_sample)
                temp_label = [0] * nratio + [1]
                temp_id = list(range(nratio + 1))
                random.shuffle(temp_id)

                shuffle_sample = []
                shuffle_label = []
                for id in temp_id:
                    shuffle_sample.append(int(neg_sample[id]))
                    shuffle_label.append(temp_label[id])

                pos_set = list(set(train_pos) - set([pos_sample]))
                allpos = [int(p) for p in random.sample(pos_set, min(50, len(pos_set)))[:50]]
                allpos += [0] * (50 - len(allpos))
                train_news_id.append(shuffle_sample)
                train_news_label.append(shuffle_label)
                train_user_id.append(user)
                user_pos.append(allpos)
        return train_news_id, train_news_label, train_user_id, test_news_id, test_news_label, \
               test_user_id, user_pos, test_user_pos, test_pos_loc

if __name__ == '__main__':
    entry = json.load(open('parameter.json','r',encoding='utf-8'))
    user_dict,behavior,news2id = entry['user_dict'],entry['behavior'],entry['news2id']
    sess = Session(user_dict)
    train_news_id, train_news_label, train_user_id, test_news_id,\
    test_news_label,test_user_id, user_pos, test_user_pos, \
    test_pos_loc = sess.generate_data(behavior,news2id,4)
    json.dump(dict(train_news_id=train_news_id,train_news_label=train_news_label,
                   train_user_id=train_user_id,test_news_id=test_news_id,test_news_label=test_news_label,
                   test_user_id=test_user_id,user_pos=user_pos,test_user_pos=test_user_pos,test_pos_loc=test_pos_loc),
              open('lists.json','w',encoding='utf-8'),indent=2)