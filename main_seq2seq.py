import torch
from torch.utils.data import TensorDataset
from utils import clean_text, config
from utils import build_vocab, build_dataset, get_pretrained_embedding
from seq2seq import EncoderRNN, DecoderRNN, training
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)
# 清洗文本
cleaned_news, cleaned_summaries = clean_text(config['text_path'], config['stopwords_path'])

# 建立词典
vocab = build_vocab(cleaned_news, cleaned_summaries, min_freq=0)
print('vocab:', vocab)
with open('E:/NLP/bighomework/Auto_Text_Summary-master/Seq2Seq/vocab_list.pkl', 'wb') as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)

# 生成 dataset 是DataTensor 格式
news_dataset = build_dataset(vocab, cleaned_news, config['max_len_news'], type='news')
#print(news_dataset)
summaries_dataset = build_dataset(vocab, cleaned_summaries, config['max_len_summaries'], type='summaries')
#print(summaries_dataset)
# 合并在一起
dataset = TensorDataset(news_dataset, summaries_dataset)

# 加载预训练的word2vec模型（使用搜狗新闻训练得到的word2vec），维度是300
pre_embeddings = get_pretrained_embedding(config['pretrained_vector_path'], vocab, vector_dim=100).to(device)

# 构建模型，选择隐状态和词向量维度相同，都是300
vocab_size = len(vocab)
print('vocabsize:', vocab_size)
# encoder 使用的是单层双向gru
encoder = EncoderRNN(vocab_size, 100, 100, n_layers=1, pre_embeddings=pre_embeddings)
#encoder = EncoderRNN(vocab_size, 300, 300, n_layers=1, use_pretrained_embeddings=False)
# decoder 使用双层单项gru
decoder = DecoderRNN(vocab_size, 100, 100, n_layers=2, pre_embeddings=pre_embeddings)
#decoder = DecoderRNN(vocab_size, 300, 300, n_layers=2, use_pretrained_embeddings=False)

# 迁移到cuda上，training 要用
encoder.to(device)
decoder.to(device)

# 训练模型
training(encoder, decoder, dataset, vocab, config['lr'], config['batch_size'], config['epochs'])






