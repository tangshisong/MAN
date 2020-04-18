from util import load_parameter,load_lists
import json
import numpy as np

train_news_id,train_news_label,train_user_id,\
test_news_id,test_news_label,test_user_id,\
user_pos,test_user_pos,test_pos_loc = load_lists('lists.json')
print("文件lists.json加载完毕。\n")
word2id,news_title,news_key = load_parameter('parameter.json')
print("文件parameters.json加载完毕。\n")

train_news_id = np.array(train_news_id,dtype='int32')
train_news_label = np.array(train_news_label,dtype='int32')
train_user_id = np.array(train_user_id,dtype='int32')

test_news_id = np.array(test_news_id,dtype='int32')
test_news_label = np.array(test_news_label,dtype='int32')
test_user_id = np.array(test_user_id,dtype='int32')
test_pos_loc = np.array(test_pos_loc,dtype='int32')

user_pos = np.array(user_pos,dtype='int32')
test_user_pos = np.array(test_user_pos,dtype='int32')

news_title = np.array(news_title,dtype='int32')
news_key = np.array(news_key,dtype='int32')
print("load.py文件执行完毕")