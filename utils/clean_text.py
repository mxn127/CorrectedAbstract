import pandas as pd
import jieba
import re

# 定义分词，sentence 是一个新闻或者摘要
def tokenizer(sentence):
    return [token for token in jieba.cut(sentence)]

# 获取停用词
def get_stopwords(path):
    with open(path, 'r', encoding='utf-8')as f:
        words = f.readlines()
        return [word.strip() for word in words]

# text 是所有新闻或者摘要，是以列表的形式存储，每一个元素是一个str形新闻或者摘要
def clean_text(text_path, stopwords_path):
    result = []
    fp = open(text_path, encoding='utf-8')
    flag = 0
    for line in fp:
        line = line.strip()  # or strip
        c = re.sub('<[^<]+?>', '', line).replace('\n', '').strip()
        if c is not '':
            if (flag % 3) == 0:
                result.append(c)
        flag += 1
        print(flag)

    news = []
    summaries = []
    flag = 0
    for e in result:
        if flag % 2 == 0:
            summaries.append(e)
        else:
            news.append(e)
        flag += 1
    #stopwords = get_stopwords(stopwords_path)
    stopwords = []
    processed_news = processed_text(news, stopwords)
    processed_summaries = processed_text(summaries, stopwords)
    #print(processed_news, processed_summaries)
    return processed_news, processed_summaries


def clean_one_text(text, stopwords_path):
    result = []
    c = re.sub('<[^<]+?>', '', text).replace('\n', '').strip()
    result.append(c)
    #stopwords = get_stopwords(stopwords_path)
    stopwords = []
    cleantext = processed_text(result, stopwords)
    return cleantext


def processed_text(text, stopwords):
    # text_tokens 是所有新闻或者摘要分词后存储的列表，其中元素是列表，表示一个新闻或者摘要
    text_tokens = []
    for sentence in text:
        tokens = tokenizer(sentence)
        tokens = [token for token in tokens if token not in stopwords]
        text_tokens.append(tokens)
    return text_tokens

def clea_text(text_path, stopwords_path):
    file = pd.read_csv(text_path)

    news = file.news.values
    summaries = file.summries.values

    stopwords = get_stopwords(stopwords_path)

    processed_news = processed_text(news, stopwords)
    processed_summaries = processed_text(summaries, stopwords)

    return processed_news, processed_summaries


if __name__ ==  "__main__":
    from config import config
    news, summs = clean_text(config['text_path'], config['stopwords_path'])
    print(news)
