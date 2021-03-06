import os
import torch
from torch import nn
import torch.utils.data as Data
import xlwt
path = "E:/NLP/大作业/Auto_Text_Summary-master/Seq2Seq"
workbook = xlwt.Workbook(encoding='utf-8')
# 创建一个worksheet
worksheet = workbook.add_sheet('My Worksheet', cell_overwrite_ok=True)


PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def batch_loss(encoder, decoder, X, Y, vocab, loss):
    """
    :param X: 是encoder端的输入数据，对应于原文，已经转移到cuda上了
    :param Y: 是decoder端的输入数据，对应于摘要
    :param loss: 是损失函数，是后面定义的nn.CrossEntropyLoss()
    """
    batch_size = X.shape[0]
    #print('X:', X[0])
    #print('Y:',Y)
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
    mask, num_not_pad_tokens = torch.ones(batch_size,).to(device), 0
    #迁移到cuda上
    l = torch.tensor([0.0]).to(device)
    for y in Y.permute(1, 0):  # Y shape: (batch, seq_len)
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        #print('dec_output:', dec_output[0])
        #print(dec_output.shape[0])
        #print('dec_state:', dec_state[0])
        #print('y:', y)
        #print('loss:', loss(dec_output, y))
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # 使用强制教学
        num_not_pad_tokens += mask.sum().item()
        # 将PAD对应位置的掩码设成0, 原文这里是 y != out_vocab.stoi[EOS], 感觉有误
        mask = mask * (y != vocab[PAD]).float()
    return l / num_not_pad_tokens



def training(encoder, decoder, dataset, vocab, lr, batch_size, num_epochs):
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    #print(data_iter)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            #print('X:', X[0])
            #print('Y:', Y[0])
            # 迁移到cuda
            X = X.to(device)
            Y = Y.to(device)
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, vocab, loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
        worksheet.write(epoch, 0, l_sum / len(data_iter))
        workbook.save('loss.xls')
        if (epoch + 1) % 10 == 0:
            #print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
            # 因为要运行很久，所以每个epoch 保存一次模型
            path = './directory/to/save/'
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(encoder.state_dict(), path+'encoder.pth')
            torch.save(decoder.state_dict(), path+'decoder.pth')
            # 加载模型在main 函数中使用：
            # encoder.load_state_dict(torch.load('./directory/to/save/encoder.pth'))
            # decoder.load_state_dict(torch.load('./directory/to/save/decoder.pth'))




