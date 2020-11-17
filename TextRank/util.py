# -*- encoding:utf-8 -*-
import jieba
import math
import numpy as np
import jieba.posseg as pseg

sentence_delimiters = frozenset('，。？；！：')
allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']


def cut_sentences(sentence):
    tmp = []
    for ch in sentence:  # 遍历字符串中的每一个字
        tmp.append(ch)
        if ch in sentence_delimiters:
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)


def cut_filter_words(cut_sentences, stopwords, use_stopwords=False):
    sentences = []
    sents = []
    for sent in cut_sentences:
        sentences.append(sent)
        if use_stopwords:
            sents.append([word for word in jieba.cut(sent) if word and word not in stopwords])  # 把句子分成词语
        else:
            sents.append([word for word in jieba.cut(sent) if word])
    return sentences, sents


def weight_map_rank(weight_graph, max_iter, tol):
    """
    输入相似度的图（矩阵)
    返回各个句子的分数
    """
    # 初始分数设置为0.5
    # 初始化每个句子的分数和老分数
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]
    denominator = calculate_degree(weight_graph)

    # 开始迭代
    count = 0
    while different(scores, old_scores, tol):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        # 计算每个句子的分数
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(old_scores, weight_graph, denominator, i)
        count += 1
        if count > max_iter:
            break
    return scores


def calculate_degree(weight_graph):
    length = len(weight_graph)
    denominator = [0.0 for _ in range(len(weight_graph))]
    for j in range(length):
        for k in range(length):
            denominator[j] += weight_graph[j][k]
        if denominator[j] == 0:
            denominator[j] = 1.0
    return denominator


def calculate_score(scores, weight_graph, denominator, i):  # i表示第i个句子
    """
    计算句子在图中的分数
    """
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        fraction = 0.0
        # 计算分子
        # [j,i]是指句子j指向句子i
        fraction = weight_graph[j][i] * scores[j]
        # 除以j的出度
        added_score += fraction / denominator[j]
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score


def different(scores, old_scores, tol=0.0001):
    """
    判断前后分数有无变化
    """
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= tol:  # 原始是0.0001
            flag = True
            break
    return flag


def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度
    """
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


def two_sentences_similarity(sents_1, sents_2):
    """
    计算两个句子的相似性
    """
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    if counter == 0:
        return 0
    return counter / (math.log(len(sents_1) + len(sents_2)))
