from gensim.models.word2vec import Word2Vec

PATH_RAW_TEXT = '../../data_0512/vocab/w2v_raw.txt'
VECTOR_SIZE = 2000

file = open(PATH_RAW_TEXT, encoding='utf-8')
sss=[]
while True:
    ss=file.readline().replace('\n','').rstrip()
    if ss=='':
        break
    s1=ss.split(" ")
    sss.append(s1)
file.close()

model = Word2Vec(vector_size=VECTOR_SIZE, workers=5, sg=1)  # 生成词向量为200维，考虑上下5个单词共10个单词，采用sg=1的方法也就是skip-gram
model.build_vocab(sss)
model.train(sss, total_examples=model.corpus_count, epochs=5)
model.save(f'./w2v_{VECTOR_SIZE}.model') 