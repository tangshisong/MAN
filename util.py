import io

import pandas as pd
from keras.layers import *
from keras.optimizers import *
import json
import numpy as np


punc = [',','.',':',';','!','?','-','"']
categoty = {'nyheter':1,'bil':2,'abonnement':3,'meninger':4,'pluss':5,'vaeret':6,'student':7,
'100sport':8,'bedriftsannonser':9,'migration catalog':10,'tjenester':11,'bolig':12,'forbruker':13,
'streaming':14,'tema':15,'omadresseavisen':16,'sport':17,'kultur':18}
DEVICE = 'CPU'

def get_embedding(file_path,word2id):
    print("开始加载词向量....")
    fin = io.open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, embed_size = map(int, fin.readline().split())
    word2vec = dict()
    for line in fin:
        tokens = line.rstrip().split(' ')
        if tokens[0] in word2id:
            word2vec[tokens[0]] = list(map(float, tokens[1:]))
    print("词表的数量：{x}；获得词向量的词个数：{y}".format(x=len(word2id),y=len(word2vec)))
    embed_matrix = [0]*len(word2id)
    can = []
    for word,vec in word2vec.items():
        embed_matrix[word2id[word]] = np.array(vec,dtype='float32')
        can.append(embed_matrix[word2id[word]])

    can = np.array(can,dtype='float32')
    mean = np.mean(can,axis=0)
    sigma = np.cov(can.T)
    norm = np.random.multivariate_normal(mean, sigma, 1)
    for i in range(len(embed_matrix)):
        if type(embed_matrix[i])==int:
            embed_matrix[i]=np.reshape(norm, embed_size)
    embed_matrix[0]=np.zeros(embed_size,dtype='float32')
    embed_matrix=np.array(embed_matrix,dtype='float32')
    # print(embed_matrix.shape)
    print("词向量加载完毕。")
    return embed_matrix

def concat(path1,path2):
    df1 = pd.read_table("path1",names=['eventid','time','userid','sessionstart','sessionstop'],
                    header=None,
                    dtype={'time':str,'sessionstart':str,'sessionstop':str})
    df2 = pd.read_table("path2",
                    names=['eventid','time','newsid','title','category','sessionstart','sessionstop','keyword'],
                    header=None)
    out = pd.merge(df1,df2,how='right',left_on='eventid',right_on='eventid')
    out.to_csv('mydata', sep='\t', index=False,na_rep='null',
           columns=['eventid','time_x','userid','sessionstart_x',
                    'sessionstop_x','newsid','title','category','keyword'])


def generate_batch_data_train(train_news_id, train_news_label, train_user_id, user_pos,
                              news_title, news_key, batch_size):
    inputid = np.arange(len(train_news_label))
    np.random.shuffle(inputid)
    y = train_news_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            candidate = news_title[train_news_id[i]]
            candidate_split = [candidate[:, k, :] for k in range(candidate.shape[1])]
            #candidate_split_mask = tf.equal(candidate_split, 0)
            candidate_body = news_key[train_news_id[i]]
            candidate_body_split = [candidate_body[:, k, :] for k in range(candidate_body.shape[1])]
            #candidate_body_split_mask = tf.equal(candidate_body_split,0)

            browsed_news = news_title[user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            #browsed_news_mask = tf.equal(browsed_news_split, 0)
            browsed_news_body = news_key[user_pos[i]]
            browsed_news_body_split = [browsed_news_body[:, k, :] for k in range(browsed_news_body.shape[1])]
            #browsed_news_body_split_mask = tf.equal(browsed_news_body_split,0)
            label = train_news_label[i]

            yield (candidate_split +
                   browsed_news_split +
                   candidate_body_split +
                   browsed_news_body_split ,[label])


def generate_batch_data_test(test_news_id, test_news_label, test_user_id, test_user_pos,
                             news_title, news_key, batch_size):
    inputid = np.arange(len(test_news_label))
    y = test_news_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            candidate = news_title[test_news_id[i]]
            #candidate_mask = tf.equal(candidate,0)
            candidate_body = news_key[test_news_id[i]]
            #candidate_body_mask = tf.equal(candidate_body,0)

            browsed_news = news_title[test_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            #browsed_news_split_mask = tf.equal(browsed_news_split,0)

            browsed_news_body = news_key[test_user_pos[i]]
            browsed_news_body_split = [browsed_news_body[:, k, :] for k in range(browsed_news_body.shape[1])]
            #browsed_news_body_split_mask = tf.equal(browsed_news_body_split,0)

            label = test_news_label[i]
            yield ([candidate] +
                   browsed_news_split +
                   [candidate_body] +
                   browsed_news_body_split , [label])

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


def load_parameter(file_path):
    entry = json.load(open(file_path, 'r', encoding='utf-8'))
    return entry['word2id'],entry['news_title'],entry['news_key']

def load_lists(file_path):
    entry = json.load(open(file_path,'r',encoding='utf-8'))
    return entry['train_news_id'],entry['train_news_label'],entry['train_user_id'],\
           entry['test_news_id'],entry['test_news_label'],entry['test_user_id'],\
           entry['user_pos'],entry['test_user_pos'],entry['test_pos_loc']

def float2int(t):
    idx = np.average(t,weights=t)
    return [1 if t[i]>=idx else 0 for i in range(len(t))]
