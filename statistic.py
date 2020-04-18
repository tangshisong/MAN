import json

x = []
userid_dict={}
news_dict={}
sample= 0
for user in open('newdata10.json',encoding='utf-8'):
    line = user.strip().split('\t')
    userid = line[2]
    sample += 1
    if line[5] not in news_dict:
        news_dict[line[5]] = 1
    else:
        news_dict[line[5]] += 1
    if userid not in userid_dict:
        userid_dict[userid] = 1
    else:
        userid_dict[userid] += 1
x = [v for k,v in userid_dict.items()]
max_c = min_c = max_m =max_x=0
for k in x:
    if k > 50:
        max_m += 1
    elif k >= 30 and k <= 50:
        max_x += 1
    elif k >= 20 and k < 30:
        max_c += 1
    else:
        min_c += 1
print('一共有记录：{}'.format(sample))
print('一共有新闻：{}条'.format(len(news_dict)))
print('用户观看新闻数>50:{}'.format(max_m))
print('用户观看新闻数[30,50]:{}'.format(max_x))
print('用户观看新闻数[20,30):{}'.format(max_c))
print('用户观看新闻数[1,20):{}'.format(min_c))
print(min(x))
print(max(x))



u = json.load(open('user.json', 'r',encoding='utf-8'))['user_dict']


f = open('test.json','a',encoding='utf-8')
for line in open('newdata10.json',encoding='utf-8'):
    news = line.strip().split('\t')
    if news[2] in u:
    #if userid_dict[news[2]] >= 10 and news[5] != 'd586d564360b83bcf5c132e3e9df8bfe60f60a9d'and news[2] not in G:
        f.write('\t'.join(news))
        f.write('\n')
