import tensorflow as tf
import matplotlib.pyplot as plt
import SimplDenseLayr
import numpy as np

import sys
sys.path.append("C:\\Users\\anujchopra\\PycharmProjects")

from DCGAN import utils
#from DCGAN import DenseLayr

st = tf.contrib.bayesflow.stochastic_tensor
normal = tf.contrib.distributions.Normal
bernaulli = tf.contrib.distributions.Bernoulli

print ("anuj")


class VariationalAe:
    def __init__(self, D, hidden_l_sizes):

        self.X = tf.placeholder(tf.float32, shape = (None,D))
        self.encoder_layers = []

        m_in = D
        for m_out in hidden_l_sizes[:-1]:
            h = SimplDenseLayr.DenseLayer(m_in,m_out)
            self.encoder_layers.append(h)
            m_in = m_out

        m_final = hidden_l_sizes[-1]

        h = SimplDenseLayr.DenseLayer(m_in,2*m_final,f = lambda x:x)
        self.encoder_layers.append(h)

        current_value = self.X

        for layer in self.encoder_layers:
            current_value = layer.forward(current_value)

        self.means = current_value[:,:m_final]
        self.dev = tf.nn.softplus(current_value[:,m_final:]) + 1e-6

        with st.value_type(st.SampleValue()):
            self.Z = st.StochasticTensor(normal(loc= self.means, scale= self.dev))

        self.decoder_layers = []
        m_in = m_final
        for m_out in reversed(hidden_l_sizes[:-1]):
            h = SimplDenseLayr.DenseLayer(m_in,m_out)
            self.decoder_layers.append(h)
            m_in = m_out

        h = SimplDenseLayr.DenseLayer(m_in,D, f = lambda x:x)
        self.decoder_layers.append(h)

        current_value = self.Z
        for layer in self.decoder_layers:
            current_value = layer.forward(current_value)

        logits = current_value
        posterior_p_logits = logits

        self.X_hat_dist = bernaulli(logits = logits)
        self.posterior_p = self.X_hat_dist.sample()
        self.posterior_p_probs = tf.nn.sigmoid(logits)

        standard_normal = normal(loc = np.zeros(m_final, np.float32), scale = np.ones(m_final, np.float32))

        Z_std = standard_normal.sample(1)
        current_value = Z_std
        for layer in self.decoder_layers:
            current_value = layer.forward(current_value)

        logits = current_value

        prior_p_dist = bernaulli(logits = logits)
        self.prior_pred = prior_p_dist.sample()
        self.prior_pred_prob = tf.nn.sigmoid(logits)

        self.Z_in = tf.placeholder(tf.float32, shape = (None,m_final))
        current_value = self.Z_in
        for layer in self.decoder_layers:
            current_value = layer.forward(current_value)

        logits = current_value
        self.prior_p_from_input_vals = tf.nn.sigmoid(logits)

        kl = tf.reduce_sum(tf.contrib.distributions.kl_divergence(self.Z.distribution, standard_normal),1)
        ell = tf.reduce_sum(self.X_hat_dist.log_prob(self.X),1)
        self.elbo = tf.reduce_sum(ell - kl)

        self.train = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(-self.elbo)
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

    def posterior_predictive_sample(self,X):
        return self.sess.run(self.posterior_p, feed_dict={self.X:X})


    def fit(self,X,epochs = 3, batch_sz = 64):
        costs = []
        n_baches = len(X)//batch_sz
        print("n_batches", n_baches)
        for i in range(epochs):
            print ("epoch:", i)
            np.random.shuffle(X)
            for j in range(n_baches):
                batch = X[j*batch_sz: (j+1)*batch_sz]
                _,c = self.sess.run((self.train, self.elbo), feed_dict={self.X:batch})
                costs.append(c)
                if j%100 == 0:
                    print ("iteration", j,"\t cost",c)

        plt.plot(costs)
        plt.show()



def main():
    X,Y = utils.get_mnist()
    X = (X > 0.5)

    vae = VariationalAe(784,[200,100])
    vae.fit(X)

    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        im = vae.posterior_predictive_sample([x]).reshape(28,28)
        plt.subplot(1,2,1)
        plt.imshow(x.reshape(28,28), cmap="gray")
        plt.title("original")
        plt.subplot(1,2,2)
        plt.imshow(im, cmap="gray")
        plt.title("generated")
        plt.show()

        ans = input("Generate another?")
        if ans and ans[0] in ["n" or "N"]:
            done = True


main()
