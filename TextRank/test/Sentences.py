from TextRank import TextRank
import os


def text_summarization(text, n):
    """
    :param text: 输入的文本字符串
    :param n: 摘要句数
    :return: 摘要句子列表
    """
    path = os.path.dirname(os.path.realpath(__file__))
    mod = TextRank.TextRank4Sentence(use_stopword=True, use_w2v=True, dict_path=path, tol=0.0001)
    summarization = mod.summarize(text, n)
    return summarization
