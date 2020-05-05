import tensorflow as tf


class bilstm_model:
    def __init__(self, vocab_size, word_dim, cell_unit, embedding=None):
        '''

        :param vocab_size:
        :param word_dim:
        :param embedding:
        '''
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.cell_unit = cell_unit
        self.word_embedding = tf.Variable(embedding, dtype=tf.float32, trainable=False)
        self.text_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="text_input")
        self.text_length = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.label_output = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.traing_tag = tf.placeholder(dtype=tf.bool)

    def embedding_layer(self, inputs, name="embedding_layer"):
        with tf.name_scope("embedding"):
            if self.word_embedding:
                embedding = tf.nn.embedding_lookup(self.word_embedding, inputs, name=name)
            else:
                embedding = tf.Variable(tf.random_normal(shape=(self.vocab_size, self.word_dim), mean=0, stddev=0.1),
                                        dtype=tf.float32,
                                        trainable=False)
            return embedding

    def bilstm_layer(self, inputs, lenghs, name="bilstm_layer"):
        cell_fw = tf.nn.rnn_cell.LSTMCell(self.cell_unit)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_fw, inputs,
                                                                 sequence_length=lenghs, dtype=tf.float32,
                                                                 name=name)
        return outputs, output_states

    def dropout_layer(self, inputs, drop_rate=0.1, name="dropout_layer"):
        return tf.layers.dropout(inputs, drop_rate, training=self.traing_tag, name=name)

    def dense_layer(self, inputs, unit, name="dense_layer"):
        with tf.name_scope(name):
            out = tf.layers.dense(inputs, unit)
            return out

    def build_model(self):
        pass

    pass
