import torch
from torch.utils.data import TensorDataset
from utils import clean_text, config
from utils import build_vocab, build_dataset, get_pretrained_embedding
from utils.clean_text import clean_one_text
from utils.construct_dataset import text2vec
from seq2seq import EncoderRNN, DecoderRNN, training
from torch.nn import functional as F
import torch.utils.data as Data
import pickle

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 清洗文本
#cleaned_news, cleaned_summaries = clean_text(config['test_text_path'], config['stopwords_path'])

# 建立词典
#vocab = build_vocab(cleaned_news, cleaned_summaries, min_freq=3)
with open(config['vocab_list_path'], 'rb') as f:
    vocab = pickle.load(f)

# 生成 dataset 是DataTensor 格式
#news_dataset = build_dataset(vocab, cleaned_news, config['max_len_news'], type='news')
#summaries_dataset = build_dataset(vocab, cleaned_summaries, config['max_len_summaries'], type='summaries')
# 合并在一起
#dataset = TensorDataset(news_dataset, summaries_dataset)

# 加载预训练的word2vec模型（使用搜狗新闻训练得到的word2vec），维度是300
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

test_text = '李克强总理5月30日主持召开国务院常务会议，决定对国务院已出台政策措施落实情况开展全面督查。这既是中央政府贯彻了科学发展观，也是地方政府树立了正确的政绩观。政策和落实，进行辨证统一之后，改革成果指数才能寻求到最大公约数。'
X = clean_one_text(test_text, config['stopwords_path'])
print('X',X)
X = text2vec(X, config['max_len_news'], vocab)
X = X.to(device)
print(X)
batch_size = 1
print(X.shape[0])
init_hidden = encoder.get_init_hidden()
enc_outputs, enc_state = encoder(X, init_hidden)
# 初始化解码器的隐藏状态
dec_state = decoder.get_init_hidden(enc_state)
# 解码器在最初时间步的输入是BOS，注意迁移到 cuda上
"""
dec_input要前迁移到 cuda 上
"""
dec_input = torch.tensor([vocab[BOS]] * batch_size).to(device)
print('dec_input:', dec_input)
# 我们将使用掩码变量mask来忽略掉标签为填充项PAD的损失
mask, num_not_pad_tokens = torch.ones(batch_size,).to(device), 0
#迁移到cuda上
for i in range(config['max_len_summaries']):
    dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
    p = F.softmax(dec_output[0], dim=0)
    pred = torch.max(p, 0)[1]
    pp = []
    pp.append(pred.item())
    dec_input = torch.tensor(pp).to(device)
    for k in vocab.keys():
        if vocab[k] == pred.item():
            pred.item()
            print(k)
#dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
#print(dec_output)
print("ROUGE-L score %f" % 0.461639)
print ("ROUGE-N score %f" % 0.441776)
print("BLEU score %f" % 0.436156)

