#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import keras
import json
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import *
from util import generate_batch_data_train,generate_batch_data_test
from util import dcg_score,ndcg_score,mrr_score,get_embedding,float2int
from load import train_news_id, train_news_label, train_user_id, \
    test_news_id, test_news_label,test_user_id, user_pos, test_user_pos, test_pos_loc
from load import word2id,news_key,news_title
keras.backend.clear_session()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MAX_SENT_LENGTH = 10
MAX_KEY_LENGTH = 10
MAX_SENTS = 50
HEAD = 16
HEAD_DIM = 16
npratio = 4
results = []

title_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
key_input = Input(shape=(MAX_KEY_LENGTH,), dtype='int32')

embedding_mat = get_embedding("cc.no.300.vec",word2id)
embedding_layer = Embedding(len(word2id), 300, weights=[embedding_mat], trainable=True)
embedded_sequences_title = embedding_layer(title_input)
embedded_sequences_title = Dropout(0.2)(embedded_sequences_title)
embedded_sequences_body = embedding_layer(key_input)
embedded_sequences_body = Dropout(0.2)(embedded_sequences_body)


title_cnn =Conv1D(padding="same", activation="relu", strides=1, filters=400, kernel_size=3)(embedded_sequences_title)
title_cnn = Dropout(0.2)(title_cnn)
'''
title_cnn = embedded_sequences_title

title_q = Dense(HEAD*HEAD_DIM)(title_cnn) #(,len,head*headdim)
title_k = Dense(HEAD*HEAD_DIM)(title_cnn)
title_v = Dense(HEAD*HEAD_DIM)(title_cnn)

title_q_heads = Reshape((MAX_SENT_LENGTH,HEAD,HEAD_DIM))(title_q)#K.reshape(title_q,(-1,K.shape(title_q)[1],HEAD,HEAD_DIM))
title_v_heads = Reshape((MAX_SENT_LENGTH,HEAD,HEAD_DIM))(title_k)#K.reshape(title_q,(-1,K.shape(title_v)[1],HEAD,HEAD_DIM))
title_k_heads = Reshape((MAX_SENT_LENGTH,HEAD,HEAD_DIM))(title_v)#K.reshape(title_q,(-1,K.shape(title_k)[1],HEAD,HEAD_DIM))

title_att = Lambda(lambda x:K.softmax(tf.einsum('bjhd,bkhd->bhjk',x[0],x[1])/(HEAD_DIM**0.5)))([title_q_heads,title_k_heads])#
title_a = Lambda(lambda x:tf.einsum('bhjk,bkhd->bjhd',x[0],x[1]))([title_att,title_v_heads])
title_a = Reshape((MAX_SENT_LENGTH,HEAD*HEAD_DIM))(title_a)#K.reshape(title_a,(-1,K.shape(title_v_heads)[1],HEAD*HEAD_DIM))
title_mul = Dense(300)(title_a)
title_mul = Dropout(0.2)(title_mul)
'''
title_mul =Dense(300)(title_cnn)
attention = Dense(200, activation='tanh')(title_mul)
attention = Flatten()(Dense(1)(attention))
attention_weight = Activation('softmax')(attention)
title_rep = keras.layers.Dot((1, 1))([title_mul, attention_weight])


key_cnn = Conv1D(padding="same", activation="relu", strides=1, filters=400, kernel_size=3)(embedded_sequences_body)
key_cnn = Dropout(0.2)(key_cnn)
'''
key_cnn = embedded_sequences_body
key_q = Dense(HEAD*HEAD_DIM)(key_cnn) #(,len,head*headdim)
key_k = Dense(HEAD*HEAD_DIM)(key_cnn)
key_v = Dense(HEAD*HEAD_DIM)(key_cnn)

key_q_heads = Reshape((MAX_KEY_LENGTH,HEAD,HEAD_DIM))(key_q)#K.reshape(key_q,(-1,K.shape(key_q)[1],HEAD,HEAD_DIM))
key_v_heads = Reshape((MAX_KEY_LENGTH,HEAD,HEAD_DIM))(key_v)
key_k_heads = Reshape((MAX_KEY_LENGTH,HEAD,HEAD_DIM))(key_v)

key_att = Lambda(lambda x:K.softmax(tf.einsum('bjhd,bkhd->bhjk',x[0],x[1])/(HEAD_DIM**0.5)))([key_q_heads,key_k_heads])
key_a = Lambda(lambda x:tf.einsum('bhjk,bkhd->bjhd',x[0],x[1]))([key_att,key_v_heads])#
key_a = Reshape((MAX_KEY_LENGTH,HEAD*HEAD_DIM))(key_a)
key_mul = Dense(300)(key_a)
#key_mul = Add()([key_mul,key_cnn])
key_mul = Dropout(0.2)(key_mul)
'''
key_mul=Dense(300)(key_cnn)
attention_key = Dense(200, activation='tanh')(key_mul)
attention_key = Flatten()(Dense(1)(attention_key))
attention_weight_key = Activation('softmax')(attention_key)
key_rep = keras.layers.Dot((1, 1))([key_mul, attention_weight_key])
all_channel = [title_rep, key_rep]
views = concatenate([Lambda(lambda x: K.expand_dims(x, axis=1))(channel) for channel in all_channel], axis=1)
attentionv = Dense(200, activation='tanh')(views)
attention_weightv = Lambda(lambda x: K.squeeze(x, axis=-1))(Dense(1)(attentionv))
attention_weightv = Activation('softmax')(attention_weightv)
newsrep = keras.layers.Dot((1, 1))([views, attention_weightv])
newsEncoder = Model([title_input, key_input], newsrep)


browsed_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
browsed_key_input = [keras.Input((MAX_KEY_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]

browsednews = [newsEncoder([browsed_news_input[_], browsed_key_input[_]]) for _ in range(MAX_SENTS)]
browsednewsrep = concatenate([Lambda(lambda x: K.expand_dims(x, axis=1))(news) for news in browsednews], axis=1)

news_q = Dense(HEAD*HEAD_DIM)(browsednewsrep) #(,len,head*headdim)
news_k = Dense(HEAD*HEAD_DIM)(browsednewsrep)
news_v = Dense(HEAD*HEAD_DIM)(browsednewsrep)

news_q_heads = Reshape((MAX_SENTS,HEAD,HEAD_DIM))(news_q)#K.reshape(key_q,(-1,K.shape(news_q)[1],HEAD,HEAD_DIM))
news_v_heads = Reshape((MAX_SENTS,HEAD,HEAD_DIM))(news_k)#K.reshape(key_q,(-1,K.shape(news_v)[1],HEAD,HEAD_DIM))
news_k_heads = Reshape((MAX_SENTS,HEAD,HEAD_DIM))(news_v)#K.reshape(key_q,(-1,K.shape(news_k)[1],HEAD,HEAD_DIM))

news_att = Lambda(lambda x:K.softmax(tf.einsum('bjhd,bkhd->bhjk',x[0],x[1])/(HEAD_DIM**0.5)))([news_q_heads,news_k_heads])
news_a = Lambda(lambda x:tf.einsum('bhjk,bkhd->bjhd',x[0],x[1]))([news_att,news_v_heads])
news_a = Reshape((MAX_SENTS,HEAD*HEAD_DIM))(news_a)
news_mul = Dense(300)(news_a)
#news_mul = Add()([news_mul,browsednewsrep])
news_mul = Dropout(0.2)(news_mul)
#news_mul = LayerNormalization()(news_mul)

attentionn = Dense(200, activation='tanh')(news_mul)
attentionn = Flatten()(Dense(1)(attentionn))
attention_weightn = Activation('softmax')(attentionn)
user_rep = keras.layers.Dot((1, 1))([news_mul, attention_weightn])

candidates_title = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(1 + npratio)]

candidates_key = [keras.Input((MAX_KEY_LENGTH,), dtype='int32') for _ in range(1 + npratio)]

candidate_vecs = [newsEncoder([candidates_title[_],candidates_key[_]]) for _ in range(1 + npratio)]

logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))

model = Model(candidates_title  + browsed_news_input + candidates_key + browsed_key_input , logits)

candidate_one_title = keras.Input((MAX_SENT_LENGTH,))

candidate_one_key = keras.Input((MAX_KEY_LENGTH,))

candidate_one_vec = newsEncoder([candidate_one_title,candidate_one_key])

score = keras.layers.Activation(keras.activations.sigmoid)(keras.layers.dot([user_rep, candidate_one_vec], axes=-1))
model_test = keras.Model([candidate_one_title]
                         + browsed_news_input
                         + [candidate_one_key]
                         + browsed_key_input, score)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

for ep in range(5):
    print("第{}轮训练开始...".format(ep))
    traingen = generate_batch_data_train(train_news_id, train_news_label, train_user_id, user_pos,news_title, news_key, 128)
    model.fit_generator(traingen, epochs=1, steps_per_epoch=len(train_news_id) // 64)
    testgen = generate_batch_data_test(test_news_id, test_news_label, test_user_id, test_user_pos,news_title, news_key, 128)
    click_score = model_test.predict_generator(testgen, steps=len(test_user_id) // 64, verbose=1)
    from sklearn.metrics import roc_auc_score,f1_score

    all_f1 = []
    all_auc = []
    all_mrr = []
    all_ndcg = []
    all_ndcg2 = []
    for m in test_pos_loc:
        if np.sum(test_news_label[m[0]:m[1]]) != 0 and m[1] < len(click_score):
            all_f1.append(f1_score(np.array(test_news_label[m[0]:m[1]]).reshape(-1,1),
                                   np.array(float2int(click_score[m[0]:m[1],0])).reshape(-1,1)))
            all_auc.append(roc_auc_score(np.array(test_news_label[m[0]:m[1]]).reshape(-1,1),
                                         np.array(float2int(click_score[m[0]:m[1], 0])).reshape(-1,1)))
            all_mrr.append(mrr_score(np.array(test_news_label[m[0]:m[1]]),
                                     np.array(click_score[m[0]:m[1], 0])))
            all_ndcg.append(ndcg_score(np.array(test_news_label[m[0]:m[1]]),
                                       np.array(click_score[m[0]:m[1], 0]), k=5))
            all_ndcg2.append(ndcg_score(np.array(test_news_label[m[0]:m[1]]),
                                        np.array(click_score[m[0]:m[1], 0]), k=10))
    print("f1 auc  mrr  ndcg  ndcg2值分别为：")
    print(np.mean(all_f1), np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg), np.mean(all_ndcg2))
    results.append([np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg), np.mean(all_ndcg2)])

json.dump(dict(results=results),open('results.json','a',encoding='utf-8'),indent=2)
model.save_weights('newsrec.h5')