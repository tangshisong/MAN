#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.utils.data as Data

def generate_batch_data_train(train_news_id, train_news_label, train_user_id, user_pos,
                              news_title, news_key, news_category, batch_size):
    torch.manual_seed(1)
    torch_dataset = Data.TensorDataset(train_news_id,user_pos,train_news_label)
    loader = Data.DataLoader(dataset=torch_dataset,batch_size=batch_size,shuffle=True,num_workers=2)

    for x,y,z in loader:
        candidate = news_title[x]
        candidate_split = [candidate[:, k, :] for k in range(candidate.size(1))]
        candidate_key = news_key[x]
        candidate_key_split = [candidate_key[:, k, :] for k in range(candidate.size(1))]
        candidate_category = news_category[x]
        candidate_category_split = [candidate_category[:, k, :] for k in range(candidate_category.size(1))]

        browsed_news = news_title[y]
        browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
        browsed_news_key = news_key[y]
        browsed_news_key_split = [browsed_news_key[:, k, :] for k in range(browsed_news_key.size(1))]
        browsed_news_category = news_category[y]
        browsed_news_category_split = [browsed_news_category[:, k, :] for k in range(browsed_news_category.size(1))]
        label = z

        #(batch,len)
        yield (candidate_split + browsed_news_split + candidate_body_split + browsed_news_body_split
                   + candidate_vertical_split + browsed_news_vertical_split + candidate_subvertical_split + browsed_news_subvertical_split,
                   [label])


def generate_batch_data_test(all_test_pn, all_label, all_test_id, batch_size):
    inputid = np.arange(len(all_label))
    y = all_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            candidate = news_words[all_test_pn[i]]
            candidate_body = news_body[all_test_pn[i]]
            candidate_vertical = news_v[all_test_pn[i]]

            browsed_news = news_words[all_test_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            browsed_news_body = news_body[all_test_user_pos[i]]
            browsed_news_body_split = [browsed_news_body[:, k, :] for k in range(browsed_news_body.shape[1])]
            browsed_news_vertical = news_v[all_test_user_pos[i]]
            browsed_news_vertical_split = [browsed_news_vertical[:, k, :] for k in
                                           range(browsed_news_vertical.shape[1])]

            label = all_label[i]
            yield ([candidate] + browsed_news_split + [candidate_body] + browsed_news_body_split + [candidate_vertical]
                   + browsed_news_vertical_split + [candidate_subvertical] + browsed_news_subvertical_split, [yy])



def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)