import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.python.keras.layers import Input,Dense, Reshape,Softmax, Lambda
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import Model


import tensorflow_probability as tfp

tfd = tfp.distributions
# initializer
tfrni = tf.random_normal_initializer
tfci = tf.constant_initializer
tfrui = tf.random_uniform_initializer
class MDN_reg_class:
    def __init__(self,_name='mdn',_x_dim=2,_y_dim=1,_k=5,_hids=[32,32],_actv=tf.nn.tanh,
                 _sig_max=0,_SCHEDULE_SIG_MAX=False,
                 _l2_reg_coef=1e-3,_batch_size=64,_sess=None,
                 _VERBOSE=True):
        # Parse arguments
        self.name = _name
        self.x_dim = _x_dim
        self.y_dim = _y_dim
        self.k = _k
        self.hids = _hids
        self.actv = _actv
        self.sig_max = _sig_max
        self.SCHEDULE_SIG_MAX = _SCHEDULE_SIG_MAX
        self.l2_reg_coef = _l2_reg_coef
        # self.sig_rate=_sig_rate
        self.batch_size=_batch_size

        # trainable layer
        #########################################
        self.layer_hidden=self.hidden_layer()
        self.layer_phi=self.phi_network()
        self.layer_mu=self.mu_network()
        self.layer_logvar=self.logvar_network()
        #########################################
        self.sess = _sess
        self.VERBOSE = _VERBOSE
        # Build graph
        self.build_graph()
        # Check parameters
        self.check_params()
        # Initialize parameters
        self.sess.run(tf.global_variables_initializer())

    def build_graph(self):
        self.x=K.placeholder(shape=[None,self.x_dim],dtype=tf.float32)
        self.y=K.placeholder(shape=[None,self.y_dim],dtype=tf.float32)
        # inputs=Input(shape=[self.x_dim],batch_size=self.batch_size) # shape=[N, self.x_dim]
        self.layers=[]
        net=self.layer_hidden(self.x)  # shape=[N, hids[0]]
        # GMM element
        self.phi=self.layer_phi(net) # Weighted Value : [N,self.k]
        self.mu=self.layer_mu(net) # mean : [N, self.y_dim, self.k]
        self.logvar=self.layer_logvar(net) # variance : [N, self.y_dim, self.k]

        self.layers.append(self.phi)
        self.layers.append(self.mu)
        self.layers.append(self.logvar)
        self.sig_rate=K.placeholder(shape=[1,], dtype=tf.float32)
        if self.sig_max==0:
            self.var=K.exp(self.logvar)
        else:
            self.var=self.sig_max*self.sig_rate*K.sigmoid(self.logvar)

        # outputs=Lambda(self.distribution, output_shape=None, name="Mixture_density")([self.phi, self.mu, self.var])
        # print(outputs)

        Mus=tf.unstack(tf.transpose(self.mu, [2, 0, 1]))
        Vars=tf.unstack(tf.transpose(self.var, [2, 0, 1]))
        loc_scale=[[Mus[i],Vars[i]] for i in range(self.k)]
        self.cat=tfd.Categorical(probs=self.phi) #
        self.comps=[tfd.MultivariateNormalDiag(loc=loc, scale_diag=tf.sqrt(scale))
                   for loc ,scale in loc_scale]

        self.tfd_mog=tfd.Mixture(cat=self.cat, components=self.comps)
        self.log_liks=self.tfd_mog.log_prob(self.y)[0] # self.y_dim=1
        self.log_lik=K.mean(self.log_liks)

        # pi_axis=K.expand_dims(self.phi,2) #[N, self.k, 1]
        print("var : {}".format(self.var))
        print("mu : {}".format(self.mu))
        self.EVs = K.dot(self.var, self.phi)
        self.EV = K.sum(self.EVs, axis=2)
        # self.EVs=K.squeeze(K.dot(self.var,pi_axis),axis=2)
        # self.EV=K.sum(self.EVs, axis=1)
        print(K.dot(K.random_normal_variable(mean=1,scale=0.5,shape=[2,1,3]),K.random_normal_variable(mean=1,scale=0.7,shape=[2,3,1])))
        print("self.vars : {}".format(self.var))

        print("K.dot(self.phi, self,var) : {}".format(K.dot(self.var, self.phi)))
        # mu_average=K.dot(self.mu, pi_axis)
        # mu_diff_sq=K.square(self.mu-mu_average)
        mu_average=K.dot(self.mu, self.phi)
        mu_diff_sq=K.square(self.mu-mu_average)
        print("mu shape : {}".format(mu_average.shape))
        print("K.square(self.mu-mu_average) : {}".format(K.square(self.mu-mu_average)))

        print("mu_average : {}".format(mu_diff_sq))
        self.VEs=K.dot(mu_diff_sq, self.phi)
        self.VE=K.sum(self.VEs,axis=2)
        print("VE : {}".format(self.VE))
        print(self.tfd_mog.sample(1))
        self.y_sample=K.squeeze(self.tfd_mog.sample(1),0)

        # self.l2_reg=K.square(self.layer_hidden.trainable_variables)+K.square(self.layer_phi.trainable_variables)+\
        #             K.square(self.layer_mu.trainable_variables)+K.square(self.layer_logvar.trainable_variables)
        self.cost = -self.log_lik
        self.optm = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(self.cost)


    # def distribution(self,element ):
    #     """
    #     element[0] : phi
    #     element[1] : mu
    #     element[2] : var
    #     :return: sample from mixture model
    #     """
    #     Mus = tf.unstack(tf.transpose(element[1], [2, 0, 1]))
    #     Vars = tf.unstack(tf.transpose(element[2], [2, 0, 1]))
    #     loc_scale = [[Mus[i], Vars[i]] for i in range(self.k)]
    #     cat = tfd.Categorical(probs=element[0])  #
    #     comps = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=tf.sqrt(scale))
    #                   for loc, scale in loc_scale]
    #     tfd_mog=tfd.Mixture(cat=cat, components=comps)
    #     return tfd_mog.sample(1)[0]



    # def custom_loss(self):
    #     tfd_mog=self.tfd_mog
    #     layer_hidden=self.layer_hidden
    #     layer_phi=self.layer_phi
    #     layer_mu=self.layer_mu
    #     layer_logvar=self.layer_logvar
    #     coef_reg=self.l2_reg_coef
    #     def loss(y_true, y_pred):
    #         log_liks=tfd_mog.log_prob(y_true)[0]
    #         log_lik=K.mean(log_liks)
    #         l2_reg=K.square(layer_hidden)+K.square(layer_phi)+K.square(layer_mu)+K.square(layer_logvar)
    #         return -log_lik+l2_reg*coef_reg
    #     return loss

    # Check parameters
    def check_params(self):
        _g_vars = tf.global_variables()
        self.g_vars = [var for var in _g_vars if '%s/' % (self.name) in var.name]
        if self.VERBOSE:
            print("==== Global Variables ====")
        for i in range(len(self.g_vars)):
            w_name = self.g_vars[i].name
            w_shape = self.g_vars[i].get_shape().as_list()
            if self.VERBOSE:
                print("  [%02d/%d] Name:[%s] Shape:[%s]" % (i, len(self.g_vars), w_name, w_shape))
        # Print layers
        if self.VERBOSE:
            print("==== Layers ====")
            n_layers = len(self.layers)
            for i in range(n_layers):
                print("  [%0d/%d] %s %s" % (i, n_layers, self.layers[i].name, self.layers[i].shape))

    # Plot results
    def plot_result(self, _x_test, _title='MDN result', _fontsize=18,
                    _figsize=(15, 5), _wspace=0.1, _hspace=0.05, _sig_rate=1.0, _pi_th=0.0,
                    _x_train=None, _y_train=None,
                    _ylim=[-3, +3]):
        # sample
        print("##########################################")
        y_sample = self.sess.run(self.y_sample, feed_dict={self.x: _x_test, self.sig_rate: _sig_rate})  # [n x y_dim]
        print("y_sample : {}".format(y_sample.shape))
        mu, var, pi = self.sess.run([self.mu, self.var, self.phi],
                                    feed_dict={self.x: _x_test, self.sig_rate: _sig_rate})
        # plot per each output dimensions (self.y_dim)
        nr, nc = 1, self.y_dim
        if nc > 2: nc = 2  # Upper limit on the number of columns
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=_wspace, hspace=_hspace)
        fig = plt.figure(figsize=_figsize)
        fig.suptitle(_title, size=_fontsize)
        for i in range(nr * nc):  # per each dimension
            ax = plt.subplot(gs[i])
            cmap = plt.get_cmap('gist_rainbow')
            colors = [cmap(ii) for ii in np.linspace(0, 1, self.k)]
            if _x_train is not None:
                plt.plot(_x_train[:, 0], _y_train[:, i], 'k.')
            plt.plot(_x_test[:, 0], y_sample[:, i], 'rx')  # plot samples per each dimension
            for j in range(self.k):  # per each mixture, plot variance
                idx = np.where(pi[:, j] < _pi_th)[0]
                plt.fill_between(_x_test[idx, 0], mu[idx, i, j] - 2 * np.sqrt(var[idx, i, j]),
                                 mu[idx, i, j] + 2 * np.sqrt(var[idx, i, j]),
                                 facecolor='k', interpolate=True, alpha=0.05)
                idx = np.where(pi[:, j] > _pi_th)[0]
                plt.fill_between(_x_test[idx, 0], mu[idx, i, j] - 2 * np.sqrt(var[idx, i, j]),
                                 mu[idx, i, j] + 2 * np.sqrt(var[idx, i, j]),
                                 facecolor=colors[j], interpolate=True, alpha=0.3)
            for j in range(self.k):  # per each mixture, plot mu
                idx = np.where(pi[:, j] > _pi_th)[0]
                plt.plot(_x_test[:, 0], mu[:, i, j], '-', color=[0.8, 0.8, 0.8], linewidth=1)
                plt.plot(_x_test[idx, 0], mu[idx, i, j], '-', color=colors[j], linewidth=3)
            plt.xlim([_x_test.min(), _x_test.max()])
            plt.ylim(_ylim)
            plt.xlabel('Input', fontsize=13)
            plt.ylabel('Output', fontsize=13)
            plt.title('[%d]-th dimension' % (i + 1), fontsize=13)
        plt.show()

    # Plot
    def plot_variances(self, _x_test, _title='blue:Var[E[y|x]] / red:E[Var[y|x]]', _fontsize=18,
                       _figsize=(15, 5), _wspace=0.1, _hspace=0.05):
        # Plot EV and VE
        VEs, EVs = self.sess.run([self.VEs, self.EVs],
                                 feed_dict={self.x: _x_test, self.sig_rate: np.array([1.0])})  # sig_rate

        VEs = 0.1 * VEs  # scale V[E[y|x]] to match that of E[V[y|x]]
        # plot per each output dimensions (self.y_dim)
        nr, nc = 1, self.y_dim
        if nc > 2: nc = 2  # Upper limit on the number of columns
        gs = gridspec.GridSpec(nr, nc)
        gs.update(wspace=_wspace, hspace=_hspace)
        fig = plt.figure(figsize=_figsize)
        fig.suptitle(_title, size=_fontsize)
        for i in range(nr * nc):  # per each dimension
            ax = plt.subplot(gs[i])
            plt.plot(_x_test.squeeze(), VEs[:, i], 'b-')
            plt.plot(_x_test.squeeze(), EVs[:, i], 'r-')
            plt.xlim([_x_test.min(), _x_test.max()])
            plt.xlabel('Input', fontsize=13)
            plt.ylabel('Output', fontsize=13)
            plt.title('[%d]-th dimension' % (i + 1), fontsize=13)
        plt.show()

    # Train the mixture of density network
    def train(self, _x_train, _y_train, _x_test, _max_iter=10000, _batch_size=256, _pi_th=0.1,
              _SHOW_EVERY=10, _figsize=(15, 5), _ylim=[-3, +3]):
        n_train = _x_train.shape[0]
        for iter in range(_max_iter):
            iter_rate_1to0 = np.exp(-4 * ((iter + 1.0) / _max_iter) ** 2)
            iter_rate_0to1 = 1 - iter_rate_1to0
            if self.SCHEDULE_SIG_MAX:  # schedule sig_max
                sig_rate = iter_rate_0to1
            else:
                sig_rate = iter_rate_0to1
            sig_rate=np.array([sig_rate])
            r_idx = np.random.permutation(n_train)[:_batch_size]
            x_batch, y_batch = _x_train[r_idx, :], _y_train[r_idx, :]  # current batch
            # Optimize the network
            _, cost_val = self.sess.run([self.optm, self.cost],
                                        feed_dict={self.x: x_batch, self.y: y_batch,
                                                   self.sig_rate: sig_rate})
            # See progress
            if ((iter % (_max_iter // _SHOW_EVERY)) == 0) | (iter == (_max_iter - 1)):
                # Plot results
                self.plot_result(_x_test=_x_test, _x_train=_x_train, _y_train=_y_train,
                                 _sig_rate=sig_rate, _pi_th=_pi_th,
                                 _title='[%d/%d] Black dots:training data / Red crosses:samples' % (iter, _max_iter),
                                 _fontsize=18, _figsize=_figsize, _ylim=_ylim)
                self.plot_variances(_x_test=_x_test,
                                    _title='blue:Var[E[y|x]] (epistemic) / red:E[Var[y|x]] (aleatoric)',
                                    _figsize=_figsize)

                # Print-out
                print("[%03d/%d] cost:%.4f" % (iter, _max_iter, cost_val))

    # trainable layer
    def hidden_layer(self):
        model = Sequential(name="HiddenLayer")
        for  idx, hid in enumerate(self.hids):
            model.add(Dense(hid, name="hidden_layer_"+str(idx),
                            kernel_regularizer=l2(self.l2_reg_coef), batch_size=self.batch_size))

        return model

    def phi_network(self):
        model = Sequential(name="PhiLayer")
        model.add(Dense(self.k, kernel_initializer=tfrni(stddev=0.01),
                        bias_initializer=tfci(0),kernel_regularizer=l2(self.l2_reg_coef)))
        # softmax 과정
        model.add(Softmax(axis=1))
        return model

    def mu_network(self):
        model=Sequential(name="Mu")
        model.add(Dense(self.k*self.y_dim, kernel_initializer=tfrni(stddev=0.01),
                        bias_initializer=tfrui(minval=-1, maxval=1),
                        kernel_regularizer=l2(self.l2_reg_coef)))
        model.add(Reshape((self.y_dim, self.k)))
        return model

    def logvar_network(self):
        model=Sequential(name="LogVar")
        model.add(Dense(self.k*self.y_dim, kernel_initializer=tfrni(stddev=0.01),
                        bias_initializer=tfci(0), kernel_regularizer=l2(self.l2_reg_coef)))
        model.add(Reshape((self.y_dim, self.k)))
        return model




