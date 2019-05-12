import tensorflow as tf
import numpy as np


class Model(object):
    def __init__(self, embedding, vocab_size, word_dim=128):
        # shape(batchsize,None)
        self.text_input = tf.placeholder(dtype=tf.int32, shape=(None, None), name="text_input")
        self.tags_output = tf.placeholder(dtype=tf.int32, shape=(None, None), name="tags_output")
        self.text_length = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.label_output = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.vocab_size = vocab_size
        self.dim = word_dim
        self._static_embedding = tf.Variable(embedding, dtype=tf.float32, trainable=False)
        self._dynamic_embedding = tf.Variable(tf.random_normal(shape=(self.vocab_size, self.dim), mean=0, stddev=0.1),
                                              dtype=tf.float32,
                                              trainable=True)

    def _embedding_layer(self, tex_ids, name="embedding"):
        with tf.name_scope("static_embedding"):
            static_embedding = tf.nn.embedding_lookup(self._static_embedding, tex_ids, name=name)
        with tf.name_scope("dynamic_embedding"):
            dynamic_embedding = tf.nn.embedding_lookup(self._dynamic_embedding, tex_ids, name=name)
        with tf.name_scope(name):
            embedding = tf.concat([static_embedding, dynamic_embedding], axis=-1, name="concat_embedding")
        return embedding

    def _dropout_layer(self, input, drop_rate=0.1, training=True, name="dropout"):
        return tf.layers.dropout(input, drop_rate, training=training, name=name)

    def _bilstm(self, input, hidden_unit, sequence_length, name="bilstm"):
        cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_unit)
        with tf.name_scope(name):
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_fw, input,
                                                                     sequence_length=sequence_length, dtype=np.float32)
        return outputs, output_states

    def _dense(self, input, tag_num, name="output"):
        with tf.name_scope(name):
            out = tf.layers.dense(input, tag_num)
            unary = tf.nn.softmax(out, name="unary")
        return unary

    def _crf_encode(self, inputs, tag_indices, sequence_lengths, name="crf_code"):
        with tf.name_scope(name):
            log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
                inputs,
                tag_indices,
                sequence_lengths
            )
        return log_likelihood

    def _crf_loss(self):
        _log_likelihood = self._crf_encode(self.unary, self.tags_output, self.text_length)
        self.crf_loss = tf.reduce_mean(-_log_likelihood)
        return self.crf_loss

    def _crf_decode(self, input, sequence_length):
        decode_tags, best_score = tf.contrib.crf.crf_decode(
            input,
            self.transition_params,
            sequence_length
        )
        _, self.acc = tf.metrics.accuracy(decode_tags, self.tags_output)
        _, self.prec = tf.metrics.precision(self.tags_output, decode_tags)
        return decode_tags, best_score

    def build_model(self, hidden_unit, tag_num):
        self._embedding = self._embedding_layer(self.text_input)
        self.embedding = self._dropout_layer(self._embedding)
        _outputs, _output_states = self._bilstm(self.embedding, hidden_unit, self.text_length)
        output = tf.concat(_outputs, axis=-1)
        out_state = tf.concat(_output_states, axis=-1)
        self.unary = self._dense(output, tag_num)

        self._crf_loss()
        self.tags, self.tag_score = self._crf_decode(self.unary, self.text_length)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.op = optimizer.minimize(self.crf_loss)
        return self.op, self.crf_loss,

    def fit(self, batch_data, session):
        feed_dict = {
            self.text_input: batch_data["data"],
            self.tags_output: batch_data["tags"],
            self.text_length: batch_data["seg_length"]
        }
        _, loss, acc = session.run([self.op, self.crf_loss, self.acc], feed_dict=feed_dict)
        return loss, acc

    def predict(self, input, session):
        feed_dict = {
            self.text_input: input,
            self.text_length: [len(x) for x in input]

        }
        tags = session.run(self.tags, feed_dict=feed_dict)
        return tags


pass

if __name__ == '__main__':
    embedding = np.eye(5, 5)
    model = Model(embedding, vocab_size=5, word_dim=5)
    model.build_model(4, 3)
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    for i in range(0, 200):
        session.run(tf.local_variables_initializer())
        batch_data = {"data": [[0, 1, 2], [0, 1, 2]],
                      "tags": [[0, 1, 2], [0, 1, 2]],
                      "seg_length": [3, 3]
                      }

        loss, acc = model.fit(batch_data, session)
        print(loss, acc)
        tag = model.predict(np.array([[0, 1]]), session)
        print("predict ", tag)

pass
