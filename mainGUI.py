from tkinter import *
import hashlib
import time
from PIL import Image
import torch
from torch.utils.data import TensorDataset
from utils import clean_text, config
from utils import build_vocab, build_dataset, get_pretrained_embedding
from utils.produce_error import specialized_part
from utils.clean_text import clean_one_text
from utils.construct_dataset import text2vec
from seq2seq import EncoderRNN, DecoderRNN, training
from torch.nn import functional as F
import torch.utils.data as Data
import pickle
from corrector.error_corrector.bert.bert_corrector import jiuzheng, BertCorrector
from TextRank import TextRank

LOG_LINE_NUM = 0
PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 建立词典
with open(config['vocab_list_path'], 'rb') as f:
    vocab = pickle.load(f)
# 加载预训练的word2vec模型（使用搜狗新闻训练得到的word2vec），维度是100
pre_embeddings = get_pretrained_embedding(config['pretrained_vector_path'], vocab, vector_dim=100).to(device)
# 构建模型，选择隐状态和词向量维度相同，都是300
vocab_size = len(vocab)
# encoder 使用的是单层双向gru
encoder = EncoderRNN(vocab_size, 100, 100, n_layers=1, pre_embeddings=pre_embeddings)
# decoder 使用双层单项gru
decoder = DecoderRNN(vocab_size, 100, 100, n_layers=2, pre_embeddings=pre_embeddings)
encoder.load_state_dict(torch.load('E:/NLP/bighomework/Auto_Text_Summary-master/Seq2Seq/directory/to/save/encoder.pth'))
decoder.load_state_dict(torch.load('E:/NLP/bighomework/Auto_Text_Summary-master/Seq2Seq/directory/to/save/decoder.pth'))
encoder = encoder.to(device)
decoder = decoder.to(device)
d = BertCorrector()
mod = TextRank.TextRank4Sentence(use_stopword=True, use_w2v=True,
                                 dict_path=config['pretrained_vector_path'], tol=0.0001)
#d.check_corrector_initialized()

class MY_GUI():
    def __init__(self,init_window_name):
        self.init_window_name = init_window_name
        self.choosetext =True

    #设置窗口
    def set_init_window(self):
        self.init_window_name.title("引入纠错机制的中文文本摘要工具")           #窗口名
        self.init_window_name.geometry('935x630+10+10')
        #标签
        self.init_data_label = Label(self.init_window_name, text="待处理文本", font=('微软雅黑', 15))
        self.init_data_label.grid(row=0, column=0)
        self.result_data_label = Label(self.init_window_name, text="输出结果", font=('微软雅黑', 15))
        self.result_data_label.grid(row=0, column=12)
        self.log_label = Label(self.init_window_name, text="纠错后文本", font=('微软雅黑', 15))
        self.log_label.grid(row=6, column=0)
        self.choose_label = Label(self.init_window_name, text="输入文本", font=('微软雅黑', 10))
        self.choose_label.grid(row=0, column=11)
        self.button_pic1 = PhotoImage(file='E:/NLP/bighomework/Auto_Text_Summary-master/Seq2Seq/UI/button1.png')
        self.button_pic2 = PhotoImage(file='E:/NLP/bighomework/Auto_Text_Summary-master/Seq2Seq/UI/button2.png')
        self.button_pic3 = PhotoImage(file='E:/NLP/bighomework/Auto_Text_Summary-master/Seq2Seq/UI/button3.png')
        self.button_pic4 = PhotoImage(file='E:/NLP/bighomework/Auto_Text_Summary-master/Seq2Seq/UI/button4.png')
        self.button_pic5 = PhotoImage(file='E:/NLP/bighomework/Auto_Text_Summary-master/Seq2Seq/UI/pic1.gif')
        #文本框
        self.init_data_Text = Text(self.init_window_name, width=50, height=20, bd=0)  # 原始数据录入框
        self.init_data_Text.grid(row=1, column=0, rowspan=5, columnspan=10)
        self.result_data_Text = Text(self.init_window_name, width=50, height=20, bd=0)  # 处理结果展示
        self.result_data_Text.grid(row=0, column=12, rowspan=7)
        self.corrected_data_Text = Text(self.init_window_name, width=50, height=20, bd=0)  #纠错结果显示
        self.corrected_data_Text.grid(row=7, column=0, columnspan=10)
        self.pic_Text = Label(self.init_window_name, width=250, height=250, bd=0, image=self.button_pic5)  # 日志框
        self.pic_Text.grid(row=7, column=12)
        #按钮
        self.makewrong_button = Button(self.init_window_name, relief=RIDGE, image=self.button_pic4, bd=0, command=self.get_wrong_text)
        self.makewrong_button.grid(row=3, column=11)
        self.correct_button = Button(self.init_window_name, relief=RIDGE, image=self.button_pic1, bd=0, command=self.get_corrected_text)
        self.correct_button.grid(row=4, column=11)
        self.abstract_button1 = Button(self.init_window_name, relief=RIDGE, image=self.button_pic2, bd=0, command=self.get_textrank_abstract)
        self.abstract_button1.grid(row=5, column=11)
        self.abstract_button2 = Button(self.init_window_name, relief=RIDGE, image=self.button_pic3, bd=0, command=self.get_abstract)
        self.abstract_button2.grid(row=6, column=11, rowspan=1)
        self.radio1 = Radiobutton(self.init_window_name, text="待处理文本", font=('微软雅黑', 10), value=False, command=self.choose_init_text)
        self.radio1.grid(row=1, column=11)
        self.radio2 = Radiobutton(self.init_window_name, text="纠错后文本", font=('微软雅黑', 10), value=True, command=self.choose_correct_text)
        self.radio2.grid(row=2, column=11)

    def choose_correct_text(self):
        self.choosetext = False

    def choose_init_text(self):
        self.choosetext = True

    #功能函数
    def get_abstract(self):
        if self.choosetext:
            test_text = self.init_data_Text.get(1.0, END).strip().replace("\n", "")
        else:
            test_text = self.corrected_data_Text.get(1.0, END).strip().replace("\n", "")
        self.result_data_Text.delete(1.0, END)
        X = clean_one_text(test_text, config['stopwords_path'])
        X = text2vec(X, config['max_len_news'], vocab)
        X = X.to(device)
        batch_size = 1
        init_hidden = encoder.get_init_hidden()
        enc_outputs, enc_state = encoder(X, init_hidden)
        # 初始化解码器的隐藏状态
        dec_state = decoder.get_init_hidden(enc_state)
        # 解码器在最初时间步的输入是BOS，注意迁移到 cuda上
        """
        dec_input要前迁移到 cuda 上
        """
        dec_input = torch.tensor([vocab[BOS]] * batch_size).to(device)
        # 我们将使用掩码变量mask来忽略掉标签为填充项PAD的损失
        mask, num_not_pad_tokens = torch.ones(batch_size, ).to(device), 0
        # 迁移到cuda上
        for i in range(config['max_len_summaries']):
            dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
            p = F.softmax(dec_output[0], dim=0)
            pred = torch.max(p, 0)[1]
            if pred.item() < vocab_size - 3:
                pp = []
                pp.append(pred.item())
                dec_input = torch.tensor(pp).to(device)
                for k in vocab.keys():
                    if vocab[k] == pred.item():
                        #print(k)
                        self.result_data_Text.insert(END, k)
            else:
                break

    def get_wrong_text(self):
        test_text = self.init_data_Text.get(1.0, END).strip().replace("\n", "")
        self.init_data_Text.delete(1.0, END)
        self.init_data_Text.insert(END, specialized_part(test_text))


    def get_textrank_abstract(self):
        if self.choosetext:
            test_text = self.init_data_Text.get(1.0, END).strip().replace("\n", "")
        else:
            test_text = self.corrected_data_Text.get(1.0, END).strip().replace("\n", "")
        self.result_data_Text.delete(1.0, END)
        self.result_data_Text.insert(END, mod.summarize(test_text, 1))

    def get_corrected_text(self):
        test_text = self.init_data_Text.get(1.0, END).strip().replace("\n", "")
        self.corrected_data_Text.delete(1.0, END)
        self.corrected_data_Text.insert(END, jiuzheng(test_text, d))



def gui_start():
    init_window = Tk()              #实例化出一个父窗口
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()          #父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示


gui_start()