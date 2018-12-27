import logging
logging.basicConfig(level=logging.INFO)
import tensorflow as tf
import os
from argparse import ArgumentParser
import numpy as np

parser = ArgumentParser()

parser.add_argument('-input', type=str, default="Datasets/ego_facebook")
parser.add_argument('-output', type=str, default="facebook_embedz")
parser.add_argument('-embed_size', type=int, default=128)

parser.add_argument('--model', type=str, default='new', choices=['old, new'])

parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--opt', choices=['GradientDescentOptimizer', 'AdamOptimizer', 'RMSPropOptimizer'],
                    default='AdamOptimizer')
parser.add_argument('-epochs', type=int, default=500)

parser.add_argument("--n_walks", default=80, type=int)
parser.add_argument("--walk_length", default=10, type=int)

parser.add_argument("--device", default='cpu', choices=['cpu', 'gpu'])

args = parser.parse_args()

class Attention:

    def __init__(self):

        self.input_path = args.input
        self.output_path = args.output
        self.embed_size = args.embed_size
        self.model = args.model
        self.lr = args.lr
        self.n_epochs = args.epochs
        self.adjmat_np = np.load(os.path.join(self.input_path, 'adjmat.npy'))
        self.n_nodes = self.adjmat_np.shape[0]
        self.walk_length = args.walk_length
        self.n_walks = args.n_walks
        self.optimizer = getattr(tf.train, args.opt)
        self.device = args.device
        self.sess = tf.Session()

    def network(self):

        self.l_embeddings = tf.Variable(tf.truncated_normal([self.n_nodes, self.embed_size]))
        self.r_embeddings = tf.Variable(tf.truncated_normal([self.n_nodes, self.embed_size]))

        self.adjmat = tf.constant(self.adjmat_np)
        diag = tf.reduce_sum(self.adjmat, axis=1)
        self.t_mat = tf.matmul(tf.diag(diag), self.adjmat)

        t = self.t_mat

        d = tf.zeros_like(self.t_mat)

        if self.model == 'new':
            self.q = tf.Variable(tf.squeeze(tf.truncated_normal([self.walk_length, 1])))

            self.q = tf.nn.softmax(self.q)

            for i in range(self.walk_length):

                d += self.q[i] * t

                t = tf.matmul(t, t)

        elif self.model=='old':
            self.lamda = tf.Variable(tf.squeeze((tf.clip_by_value(tf.truncated_normal([1,1], 1), 1.1, 1e+10))))

            self.lamdas = []
            sum=0

            for i in range(1, self.walk_length + 1):
                lamda_i = tf.sigmoid(self.lamda) ^ i
                sum += lamda_i
                self.lamdas.append(lamda_i)

            for i in range(self.walk_length):
                self.lamdas[i] /= sum

            for i in range(self.walk_length):

                d += self.lamdas[i-1] * t

                t = tf.matmul(t, t)

        p0 = tf.eye(self.n_nodes) * self.n_walks

        d = tf.matmul(p0, d)

        self.lrt = tf.matmul(self.l_embeddings, tf.transpose(self.r_embeddings))

        self.part1 = -d * tf.log(tf.clip_by_value(tf.sigmoid(self.lrt), 1e-10, 1))

        self.part2 = -(1-self.adjmat) * tf.log(1 - tf.clip_by_value(tf.sigmoid(self.lrt), 1e-10, 1))

        self.part3 = tf.norm(self.q)

        self.loss = tf.reduce_sum(self.part1) + tf.reduce_sum(self.part2) + self.part3

    def train(self):
        logging.info(". . . Starting training . . .")

        with tf.device("/{}:0".format(args.device)):

            self.network()

            self.train_op = self.optimizer(learning_rate=self.lr).minimize(self.loss)

            self.sess.run(tf.global_variables_initializer())

            for i in range(self.n_epochs):

                self.sess.run(self.train_op)

                logging.info("epoch {}, loss is {} {} {} lrt {}".format(i,
                                                           self.sess.run(self.loss),
                                                           self.sess.run(self.part1),
                                                           self.sess.run(self.part2),
                                                           self.sess.run(self.part3)),
                                                           self.sess.run(self.lrt))

                if(i+1%100 == 0):
                    np.save(self.output_path, self.sess.run(self.l_embeddings))

        self.sess.close()

if __name__ == "__main__":
    at = Attention()

    at.train()