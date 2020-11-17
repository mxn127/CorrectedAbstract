# -*- coding: utf-8 -*-
# Author: XuMing(xuming624@qq.com)
# Brief: config

import os

pwd_path = os.path.abspath(os.path.dirname(__file__))

# -----用户目录，存储模型文件-----
USER_DATA_DIR = os.path.expanduser('~/.pycorrector/datasets')
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)
language_model_path = os.path.join(USER_DATA_DIR, 'zh_giga.no_cna_cmn.prune01244.klm')

# -----词典文件路径-----
# 通用分词词典文件  format: 词语 词频
word_freq_path = os.path.join(pwd_path, 'data/word_freq.txt')
# 中文常用字符集
common_char_path = os.path.join(pwd_path, 'data/common_char_set.txt')
# 同音字
same_pinyin_path = os.path.join(pwd_path, 'data/same_pinyin.txt')
# 形似字
same_stroke_path = os.path.join(pwd_path, 'data/same_stroke.txt')
# 用户自定义错别字混淆集  format:变体	本体   本体词词频（可省略）
custom_confusion_path = os.path.join(pwd_path, 'data/custom_confusion.txt')
# 用户自定义分词词典  format: 词语 词频
custom_word_freq_path = os.path.join(pwd_path, 'data/custom_word_freq.txt')
# 知名人名词典 format: 词语 词频
person_name_path = os.path.join(pwd_path, 'data/person_name.txt')
# 地名词典 format: 词语 词频
place_name_path = os.path.join(pwd_path, 'data/place_name.txt')
# 停用词
stopwords_path = os.path.join(pwd_path, 'data/stopwords.txt')
# 搭配词
ngram_words_path = os.path.join(pwd_path, 'data/ngram_words.txt')
# 英文文本
en_text_path = os.path.join(pwd_path, 'data/en/big.txt')
