# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: use bert detect and correct chinese char error
"""

import operator
import os
import sys
import time

from transformers import pipeline

sys.path.append('../..')
from pycorrector.utils.text_utils import is_chinese_string, convert_to_unicode
from pycorrector.utils.logger import logger
from pycorrector.corrector import Corrector

pwd_path = os.path.abspath(os.path.dirname(__file__))


class BertCorrector(Corrector):
    def __init__(self, bert_model_dir=os.path.join(pwd_path, 'data/bert_models/chinese_finetuned_lm/')):
        super(BertCorrector, self).__init__()
        self.name = 'bert_corrector'
        t1 = time.time()
        self.model = pipeline('fill-mask',
                              model=bert_model_dir,
                              tokenizer=bert_model_dir)
        if self.model:
            self.mask = self.model.tokenizer.mask_token
            logger.debug('Loaded bert model: %s, spend: %.3f s.' % (bert_model_dir, time.time() - t1))
        self._initialize_detector()

    def bert_correct(self, text):
        """
        句子纠错
        :param text: 句子文本
        :return: corrected_text, list[list], [error_word, correct_word, begin_pos, end_pos]
        """
        text_new = ''
        details = []
        #self.check_corrector_initialized()
        # 编码统一，utf-8 to unicode
        text = convert_to_unicode(text)
        # 长句切分为短句
        blocks = self.split_2_short_text(text, include_symbol=True)
        for blk, start_idx in blocks:
            blk_new = ''
            for idx, s in enumerate(blk):
                # 处理中文错误
                if is_chinese_string(s):
                    sentence_lst = list(blk_new + blk[idx:])
                    sentence_lst[idx] = self.mask
                    sentence_new = ''.join(sentence_lst)
                    # 预测，默认取top5
                    predicts = self.model(sentence_new)
                    top_tokens = []
                    for p in predicts:
                        token_id = p.get('token', 0)
                        token_str = self.model.tokenizer.convert_ids_to_tokens(token_id)
                        top_tokens.append(token_str)

                    if top_tokens and (s not in top_tokens):
                        # 取得所有可能正确的词
                        candidates = self.generate_items(s)
                        if candidates:
                            for token_str in top_tokens:
                                if token_str in candidates:
                                    details.append([s, token_str, start_idx + idx, start_idx + idx + 1])
                                    s = token_str
                                    break
                blk_new += s
            text_new += blk_new
        details = sorted(details, key=operator.itemgetter(2))
        return text_new, details


def jiuzheng(text, d):
        corrected_error, errs = d.bert_correct(text)
        return corrected_error

if __name__ == "__main__":
    corrected_error=jiuzheng()
    print(corrected_error)
'''    
    d = BertCorrector()
    error=input('输入错误句子')
    corrected_error, errs = d.bert_correct(error)
    print("original sentence:{} => {}, err:{}".format(error, corrected_error, errs))
'''
'''
    error_sentences = [
        '疝気医院那好 为老人让坐，疝気专科百科问答',
        '少先队员因该为老人让坐',
        '少 先  队 员 因 该 为 老人让坐',
        '机气学习是人工智能领遇最能体现智能的一个分知',
        '我十人',
    ]
    for sent in error_sentences:
        corrected_sent, err = d.bert_correct(sent)
        print("original sentence:{} => {}, err:{}".format(sent, corrected_sent, err))
'''
