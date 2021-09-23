from operator import index
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


class MyCorpus():
    def __init__(self) -> None:
        self.index = 0

    def __iter__(self):
        self.index = 0
        yield TaggedDocument('remove email addresses', [self.index])
        print(self.index)
        yield TaggedDocument('remove websites addresses', [self.index])
        print(self.index)
        yield TaggedDocument('remove quotes', [self.index])
        print(self.index)
        yield TaggedDocument('remove stopwords', [self.index])
        print(self.index)

model = Doc2Vec(MyCorpus(), vector_size=5, window=2, min_count=1, workers=4)
model.train()
vector = model.infer_vector(["remove", "quotes"])
model.save('test_save_doc2vec_model')
print(vector)