import random
import numpy as np


class DataSet(object):
    def __init__(self, data_path, word2id, tag2id, label2id, max_len, batch=3):
        self.batch = batch
        self.offset = 0
        self.data = []
        self.max_len = max_len
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
            query_list = [self.word2id.get(word, 0) for word in query.split(" ")]
            tags_list = [self.tag2id.get(word, 0) for word in tags.split(" ")]
            text_len = len(query_list)
            if text_len != len(tags_list):
                print("len(query_list)!=len(tags_list) " + data)
                continue
                pass
            if text_len > self.max_len:
                query_list = query_list[:self.max_len]
                tags_list = tags_list[:self.max_len]
                query_len_list.append(self.max_len)
            else:
                query_list.extend([0] * (self.max_len - text_len))
                tags_list.extend([0] * (self.max_len - text_len))
                query_len_list.append(text_len)
            label_id_list.extend([self.label2id.get(label.strip(), 1)])
            query_id_list.append(query_list)
            tag_id_list.append(tags_list)

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
    word2id = {}
    tag2id = {}
    label2id = {}
    with open("../data/data.txt", encoding="utf8") as f:
        word_list = []
        tag_list = []
        label_lit = []

        for line in f:
            query = line.split("\t")
            word_list.extend(query[0].split(" "))
            tag_list.extend(query[1].split(" "))
            label_lit.append(query[2].strip())
            pass
        for id, word in enumerate(list(set(word_list))):
            word2id[word] = id
        for id, word in enumerate(list(set(tag_list))):
            tag2id[word] = id
        for id, word in enumerate(list(set(label_lit))):
            label2id[word] = id
    print("word2id ", word2id)
    print("label2id ", label2id)
    print("tag2id ", tag2id)
    for i in range(0, 1):
        data = DataSet("../data/data.txt", word2id, tag2id, label2id, max_len=32, batch=7).shuffle()
        for item_batch in data:
            query_id_list, query_len_list, tag_id_list, label_id_list = item_batch
            print("query_id_list: ", np.array(query_id_list).shape)
            print("query_len_list: ", np.array(query_len_list).shape)
            print("tag_id_list: ", np.array(tag_id_list).shape)
            print("label_id_list: ", np.array(label_id_list).shape)
            break
