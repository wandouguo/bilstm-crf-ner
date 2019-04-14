import random
import numpy as np


class DataSet(object):
    def __init__(self, data_path, word2id, tag2id, label2id, batch=3):
        self.batch = batch
        self.offset = 0
        self.data = []
        if isinstance(word2id, dict):
            self.word2id = word2id
            self.tag2id = tag2id
            self.label2id = label2id
        else:
            raise Exception("wod2id need dict!!")
        with open(data_path, encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if line == "" or line.startswith("#"):
                    continue
                else:
                    self.data.append(line.split("\t"))
        self._init()

    def _init(self):
        self.data_length = len(self.data)
        self.end = self.data_length // self.batch
        self.rem = self.data_length % self.batch
        if self.rem != 0:
            self.end = self.end + 1
        print(len(self.data))

    def shuffle(self):
        random.shuffle(self.data)
        return self

    def repeat(self, num=1):
        self.data = self.data * num
        self._init()
        return self
        pass

    def __iter__(self):
        return self

    def gen_batch_data(self, data_list):
        query_id_list = []
        query_len_list = []
        tag_id_list = []
        label_id_list = []
        for data in data_list:
            query, tags, label = data
            query_id_list.append([self.word2id.get(word, "<PAD>") for word in query.split(" ")])
            query_len_list.append(len(query.split(" ")))
            tag_id_list.append([self.tag2id.get(word, "<PAD>") for word in tags.split(" ")])
            label_id_list.extend([self.label2id.get(label,"other")])
            pass
        return query_id_list, query_len_list, tag_id_list, label_id_list

    def __next__(self):
        if self.offset == self.end:
            print("data_finish!")
            raise StopIteration()
        start = self.offset * self.batch
        self.offset += 1
        batch_data = self.data[start:start + self.batch]
        return self.gen_batch_data(batch_data)


if __name__ == '__main__':
    # print(3 % 2)

    for i in range(0, 2):
        data = DataSet("../data/data.txt", {}, {}, {}, batch=2).shuffle()
        for item_batch in data:
            print("item", item_batch)
