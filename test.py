#!/usr/bin/env python3
# -*- coding: utf-8 -*-

with open("txs.txt","r",encoding="utf-8") as f:
    print(f.read())

'''

import random
import tensorflow as tf

a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用

b = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)                                  # 判断GPU是否可以用

print(a)
print(b)
'''

'''

word2id = load('parameter.json','word2id')
embedding_mat = get_embedding("cc.no.300.vec",word2id)


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
            candidate_split_mask = [tf.equal(x, 0) for x in candidate_split]
            candidate_body = news_key[train_news_id[i]]
            candidate_body_split = [candidate_body[:, k, :] for k in range(candidate_body.shape[1])]
            candidate_body_split_mask = [tf.equal(x,0) for x in candidate_body_split]

            browsed_news = news_title[user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            browsed_news_mask = [tf.equal(x, 0) for x in browsed_news_split]
            browsed_news_body = news_key[user_pos[i]]
            browsed_news_body_split = [browsed_news_body[:, k, :] for k in range(browsed_news_body.shape[1])]
            browsed_news_body_split_mask = [tf.equal(x, 0) for x in browsed_news_body_split]

            label = train_news_label[i]

            yield (candidate_split + candidate_split_mask+
                   browsed_news_split + browsed_news_mask+
                   candidate_body_split + candidate_body_split_mask+
                   browsed_news_body_split + browsed_news_body_split_mask,[label])


def generate_batch_data_test(test_news_id, test_news_label, test_user_id, test_user_pos,
                             news_title, news_key, batch_size):
    inputid = np.arange(len(test_news_label))
    y = test_news_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            candidate = news_title[test_news_id[i]]
            candidate_mask = tf.equal(candidate,0)
            candidate_body = news_key[test_news_id[i]]
            candidate_body_mask = tf.equal(candidate_body,0)

            browsed_news = news_title[test_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            browsed_news_split_mask = [tf.equal(x,0) for x in browsed_news_split]

            browsed_news_body = news_key[test_user_pos[i]]
            browsed_news_body_split = [browsed_news_body[:, k, :] for k in range(browsed_news_body.shape[1])]
            browsed_news_body_split_mask = [tf.equal(x,0) for x in browsed_news_body_split]

            label = test_news_label[i]
            yield ([candidate] + [candidate_mask]+
                   browsed_news_split + browsed_news_split_mask+
                   [candidate_body] + [candidate_body]+
                   browsed_news_body_split + browsed_news_body_split_mask, [label])



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MAX_SENT_LENGTH = 15
MAX_KEY_LENGTH = 15
MAX_SENTS = 50
HEAD = 16
HEAD_DIM = 16
npratio = 4
results = []

#embedding
title_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
title_mask = Input(shape=(MAX_SENT_LENGTH,),dtype='int32')
key_input = Input(shape=(MAX_KEY_LENGTH,), dtype='int32')
key_mask = Input(shape=(MAX_KEY_LENGTH,), dtype='int32')

embedding_mat = get_embedding("cc.no.300.vec",word2id)
embedding_layer = Embedding(len(word2id), 300, weights=[embedding_mat], trainable=True)
embedded_sequences_title = embedding_layer(title_input)
embedded_sequences_title = Dropout(0.2)(embedded_sequences_title)
embedded_sequences_body = embedding_layer(key_input)
embedded_sequences_body = Dropout(0.2)(embedded_sequences_body)

#title_cnn = embedded_sequences_title
title_cnn =Conv1D(padding="same", activation="relu", strides=1, filters=400, kernel_size=3)(embedded_sequences_title)
title_cnn = Dropout(0.2)(title_cnn)
#multi-head

title_mul = MultiHeadAttention(HEAD, HEAD_DIM)([title_cnn, title_cnn, title_cnn, title_mask])

attention = Dense(200, activation='tanh')(title_mul)
attention = Flatten()(Dense(1)(attention))
attention_weight = Activation('softmax')(attention)
title_rep = keras.layers.Dot((1, 1))([title_mul, attention_weight])

key_cnn = Conv1D(padding="same", activation="relu", strides=1, filters=400, kernel_size=3)(embedded_sequences_body)
key_cnn = Dropout(0.2)(key_cnn)

transform_key = [Lambda(lambda x:keras.layers.Dot()([x,K.transpose(x)]))(Dense(16)(key_cnn))
                   for head in range(HEAD)]
new_transform_key = [Lambda(lambda x:Flatten()(Dense(1)(x)))(att) for att in transform_key]
new_transform_key_weight = [Lambda(lambda x:Activation('softmax')(x))(att) for att in new_transform_key]
key_sep_rep = [Lambda(lambda x,y:keras.layers.Dot((1,1))([x,y]))(m,n)
           for a,b in zip(transform_key,new_transform_key_weight) for m,n in zip(a,b)]
combine_key_self = concatenate(key_sep_rep,axis=1)
'''
'''


#key_cnn = embedded_sequences_body
key_mul = MultiHeadAttention(HEAD,HEAD_DIM)([key_cnn,key_cnn,key_cnn,key_mask])
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
newsEncoder = Model([title_input, title_mask, key_input, key_mask], newsrep)


browsed_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
browsed_news_mask = [keras.Input(shape=(MAX_SENT_LENGTH,),dtype='int32') for _ in range(MAX_SENTS)]
browsed_key_input = [keras.Input((MAX_KEY_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
browsed_key_mask = [keras.Input((MAX_KEY_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]

browsednews = [newsEncoder([browsed_news_input[_], browsed_news_mask[_],browsed_key_input[_],browsed_key_mask[_]]) for
               _ in range(MAX_SENTS)]
browsednewsrep = concatenate([Lambda(lambda x: K.expand_dims(x, axis=1))(news) for news in browsednews], axis=1)


transform_title = [Lambda(lambda x:keras.layers.Dot()([x,K.transpose(x)]))(Dense(16)(title_cnn))
                   for head in range(HEAD)]
new_transform_title = [Lambda(lambda x:Flatten()(Dense(1)(x)))(att) for att in transform_title]
new_transform_title_weight = [Lambda(lambda x:Activation('softmax')(x))(att) for att in new_transform_title]
sep_rep = [Lambda(lambda x,y:keras.layers.Dot((1,1))([x,y]))(m,n)
           for a,b in zip(transform_title,new_transform_title_weight) for m,n in zip(a,b)]
combine_title_self = concatenate(sep_rep,axis=1)








test_neg = [11,17,9,0,12,6,13,1,5,21,100,1000,10000,111,11111]
test_pos = [99,70,9]
idx = sorted(random.sample([i for i in range(1,len(test_neg))], len(test_pos) - 1))
idx.append(len(test_neg))
temp = []
start = 0
for i in idx:
    temp.append(test_neg[start:i])
    start = i
print(temp)


BATCH_SIZE = 2
m = torch.tensor([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]])
x = torch.tensor([[1,2,3,4,5],[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
y = torch.tensor([[1,2,3,4,5],[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])
z = torch.tensor([[1,2,3,4,5],[1,1,1,1,1],[2,2,2,2,2],[3,3,3,3,3]])

torch_dataset = Data.TensorDataset(x, y,z)

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

for epoch in range(3):
    for step, (batch_x, batch_y,z) in enumerate(loader):
        if len(z)==0:
            print("no")
        xx=m[z]
        print(xx.size())
        print(m[z])
        s = [xx[:,k,:] for k in range(xx.size(1))]
        print(s)
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy(),z.numpy())

'''
