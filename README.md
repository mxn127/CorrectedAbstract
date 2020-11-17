# CorrectedAbstract
随着大数据时代的到来, 海量的信息不断地涌入到人们的生活当中，如何高效过滤精简这些信息数据成为了一个非常紧要的问题。文本摘要能够实现对文本关键信息的提取，其方法主要分为抽取式和生成式两大类，分别通过提取文档中存在的关键词和建立抽象语意表示形成自动文摘。现有的模型通常直接对文本进行关键词和摘要提取，然而以网络作为传播媒介的互联网文本信息的文本错误率要比纸质媒介传播下的文本高很多，这限制了关键词和摘要提取的准确性，因此我们在文本摘要中引入文本纠错机制。文本纠错实现对错误字词、错误语法等的识别纠正，主要分为基于统计语言模型和基于深度模型的两类方法。基于深度模型的方法通过端到端的方式减少人工特征提取，泛化能力强，使用效果好，在文本纠错中有着广泛的应用。因此，为了有效地过滤并精简信息，让人们更高效地从互联网文本中获取信息，我们的项目综合了文本摘要和文本纠错两大任务，基于textRank算法、Seq2Seq模型结构和kenlm、bert、soft-masked bert模型实现了对互联网文本信息的纠错和自动摘要，可以应用在舆情监控、数据分析，数据挖掘等多个领域。