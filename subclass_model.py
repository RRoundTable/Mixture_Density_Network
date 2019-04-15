import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.python.keras.layers import Input,Dense, Reshape,Softmax, Lambda, multiply, BatchNormalization
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


class MDN_reg_class(Model):
    def __init__(self,_x_dim=2,_y_dim=1,_k=5,_hids=[32,32],_actv=tf.nn.tanh,
                 _sig_max=0,_SCHEDULE_SIG_MAX=False,_sig_rate=1.0,
                 _l2_reg_coef=1e-3,_batch_size=64,_sess=None,
                 _VERBOSE=True):
        super(MDN_reg_class, self).__init__()

        # Parse arguments
        # self.name = _name
        self.x_dim = _x_dim
        self.y_dim = _y_dim
        self.k = _k
        self.hids=_hids
        self.actv = _actv
        self.sig_max = _sig_max
        self.SCHEDULE_SIG_MAX = _SCHEDULE_SIG_MAX
        self.l2_reg_coef = _l2_reg_coef
        self.sig_rate=_sig_rate
        self.batch_size=_batch_size

        # trainable layer
        #########################################
        self.layer_hidden=self.hidden_layer()
        self.layer_phi=self.phi_network()
        self.layer_mu=self.mu_network()
        self.layer_logvar=self.logvar_network()
        #########################################
        self.VERBOSE = _VERBOSE
        K.set_session(_sess)


    def call(self,x):
        net = self.layer_hidden(x)  # shape=[N, hids[0]]
        # GMM element
        phi = self.layer_phi(net) # Weighted Value : [N,self.k]
        mu = self.layer_mu(net) # mean : [N, self.y_dim, self.k]
        logvar = self.layer_logvar(net) # variance : [N, self.y_dim, self.k]
        if self.sig_max==0:
            var = K.exp(logvar)
        else:
            var = self.sig_max * self.sig_rate * K.sigmoid(logvar)
        var = Lambda((lambda x: x + 0.01))(var)
        # # GMM distribution
        outputs=Lambda(self.distribution, output_shape=[self.y_dim], name="Mixture_density", trainable=False)([phi, mu, var])
        return outputs

    def distribution(self,element):
        """
        element[0] : phi
        element[1] : mu
        element[2] : var
        :return: sample from mixture model
        """
        Mus = tf.unstack(tf.transpose(element[1], [2, 0, 1]))
        Vars = tf.unstack(tf.transpose(element[2], [2, 0, 1]))
        loc_scale = [[Mus[i], Vars[i]] for i in range(self.k)]
        cat = tfd.Categorical(probs=element[0])  #
        comps = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=tf.sqrt(scale))
                      for loc, scale in loc_scale]
        tfd_mog=tfd.Mixture(cat=cat, components=comps)
        return tfd_mog

    def get_Variance(self,x):
        net=self.layer_hidden(x)
        phi=self.layer_phi(net) # [N,self.k]
        mu=self.layer_mu(net) # [N, self.y_dim, self.k]
        logvar=self.layer_logvar(net)  # [N, self.y_dim, self.k]
        if self.sig_max==0:
            var = K.exp(logvar)
        else:
            var = self.sig_max*self.sig_rate*K.sigmoid(logvar)
        phi = K.expand_dims(phi,1)

        EVs = K.sum(multiply([phi,var]),axis=2)
        # EVs=K.sum(EVs,axis=1)

        mu_average = K.sum(multiply([phi,mu]),axis=2)
        mu_diff_sq = K.square(mu-K.expand_dims(mu_average,2))

        VEs = K.sum(multiply([phi,mu_diff_sq]), axis=2)
        # VEs=K.sum(VEs,axis=1)
        return K.eval(EVs), K.eval(VEs)

    def custom_loss(self, x_train,y_true):
        tfd_mog = self.call(x_train)
        log_liks = tfd_mog.log_prob(y_true)
        log_lik = tf.reduce_mean(log_liks)
        return -log_lik

    # # Plot results
    def plot_result(self, _x_test,epoch, _title='MDN result', _fontsize=18,
                    _figsize=(15, 5), _wspace=0.1, _hspace=0.05, _sig_rate=1.0, _pi_th=0.1,
                    _x_train=None, _y_train=None,
                    _ylim=[-3, +3]):
        # sample
        self.sig_rate = _sig_rate
        distribution = self.call(_x_test)
        y_sample = K.eval(K.squeeze(distribution.sample(1),0))
        net = self.layer_hidden(_x_test)
        phi = K.eval(self.layer_phi(net))  # [N,self.k]
        mu = K.eval(self.layer_mu(net))  # [N, self.y_dim, self.k]
        logvar = K.eval(self.layer_logvar(net))  # [N, self.y_dim, self.k]
        if self.sig_max == 0:
            var = K.exp(logvar)
            var = K.eval(var)
        else:
            var=self.sig_max*self.sig_rate*K.sigmoid(logvar)
            var = K.eval(var)
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
                idx = np.where(phi[:, j] < _pi_th)[0]
                if len(idx) > 0:
                    plt.fill_between(_x_test[idx, 0],
                                     mu[idx, i, j] - 2 * np.sqrt(var[idx, i, j]),
                                     mu[idx, i, j] + 2 * np.sqrt(var[idx, i, j]),
                                     facecolor='k', interpolate=True, alpha=0.05)
                idx = np.where(phi[:, j] > _pi_th)[0]
                if len(idx) > 0:
                    plt.fill_between(_x_test[idx, 0],
                                     mu[idx, i, j] - 2 * np.sqrt(var[idx, i, j]),
                                     mu[idx, i, j] + 2 * np.sqrt(var[idx, i, j]),
                                     facecolor=colors[j], interpolate=True, alpha=0.3)
            for j in range(self.k):  # per each mixture, plot mu
                idx = np.where(phi[:, j] > _pi_th)[0]
                plt.plot(_x_test[:, 0], mu[:, i, j], '-', color=[0.8, 0.8, 0.8], linewidth=1)
                plt.plot(_x_test[idx, 0], mu[idx, i, j], '-', color=colors[j], linewidth=3)
            plt.xlim([_x_test.min(), _x_test.max()])
            plt.ylim(_ylim)
            plt.xlabel('Input', fontsize=13)
            plt.ylabel('Output', fontsize=13)
            plt.title('[%d]-th dimension' % (i + 1), fontsize=13)
        plt.savefig("./result/epoch_{}.png".format(epoch))

    # Plot
    def plot_variances(self, _x_test,epoch,_title='blue:Var[E[y|x]] / red:E[Var[y|x]]', _fontsize=18,
                       _figsize=(15, 5), _wspace=0.1, _hspace=0.05):
        # Plot EV and VE
        self.sig_rate = 1.0
        EVs, VEs = self.get_Variance(_x_test)
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
        plt.savefig("./variance/epoch_{}.png".format(epoch))

    # trainable layer
    def hidden_layer(self):
        model = Sequential(name="HiddenLayer")
        for idx, hid in enumerate(self.hids):
            if idx==0:
                model.add(Dense(hid, name="hidden_layer_" + str(idx),
                                kernel_regularizer=l2(self.l2_reg_coef),
                                dtype=tf.float32,
                                input_shape=(self.batch_size,self.x_dim),
                                activation=self.actv))
                # model.add(BatchNormalization())
            else:
                model.add(Dense(hid, name="hidden_layer_"+str(idx),
                            kernel_regularizer=l2(self.l2_reg_coef),
                                dtype=tf.float32,
                                activation=self.actv))
                # model.add(BatchNormalization())

        return model

    def phi_network(self):
        model = Sequential(name="PhiLayer")
        model.add(Dense(self.k, kernel_initializer=tfrni(stddev=0.01),
                        bias_initializer=tfci(0),
                        kernel_regularizer=l2(self.l2_reg_coef),
                        dtype=tf.float32,activation=None))
        model.add(BatchNormalization())
        model.add(Softmax(axis=1))
        return model

    def mu_network(self):
        model = Sequential(name="Mu")
        model.add(Dense(self.k*self.y_dim, kernel_initializer=tfrni(stddev=0.01),
                        bias_initializer=tfrui(minval=-1, maxval=1),
                        kernel_regularizer=l2(self.l2_reg_coef),
                        dtype=tf.float32,
                        activation=None))
        model.add(BatchNormalization())
        model.add(Reshape((self.y_dim, self.k)))
        return model

    def logvar_network(self):
        model = Sequential(name="LogVar")
        model.add(Dense(self.k*self.y_dim, kernel_initializer=tfrni(stddev=0.01),
                        bias_initializer=tfci(0),
                        kernel_regularizer=l2(self.l2_reg_coef),
                        dtype=tf.float32,
                        activation=None))
        model.add(BatchNormalization())
        model.add(Reshape((self.y_dim, self.k)))
        return model




