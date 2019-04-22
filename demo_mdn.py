import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.optimizers import RMSprop, Adam
# from model import MDN_reg_class
from subclass_model import MDN_reg_class
from util import gpu_sess,nzr
from tensorflow.python.keras import backend as K
import os
from tensorflow.python.client import device_lib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-batch_size', type=int, default=256)
parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
parser.add_argument('-k', type=int, default=20, help='learning rate')
parser.add_argument('-num_epoch', type=int, default=20000)
parser.add_argument('-activation', type=str, default='tanh')
parser.add_argument('-optimizer', type=str, default='rmsp')
opt = parser.parse_args()
print(device_lib.list_local_devices())

if not os.path.exists("./result"):
    os.mkdir("./result")
if not os.path.exists("./variance"):
    os.mkdir("./variance")

tf.reset_default_graph() # Reset graph
tf.set_random_seed(seed=1)
np.random.seed(seed=1)
sess = gpu_sess()

def train(model, optimizer, x_train, y_train, x_test, epoch, batch_size):
    # placeholder
    x_shape, y_shape = list(x_train.shape), list(y_train.shape)
    x_shape[0], y_shape[0]= None, None
    _x_batch=K.placeholder(name='x', shape=x_shape)
    _y_batch=K.placeholder(name='y', shape=y_shape)
    loss = model.custom_loss(_x_batch, _y_batch)
    variables = model.variables
    updates = optimizer.get_updates(params=variables,loss=loss)
    _train=K.function([_x_batch,_y_batch],[loss],updates=updates)
    for e in range(epoch):
        n_train = x_train.shape[0]
        iter_rate_1to0 = np.exp(-4 * ((e + 1.0) / epoch) ** 2)
        iter_rate_0to1 = 1 - iter_rate_1to0
        if model.SCHEDULE_SIG_MAX:  # schedule sig_max
            model.sig_rate = iter_rate_0to1
        else:
            model.sig_rate = iter_rate_0to1
        r_idx = np.random.permutation(n_train)[:batch_size]
        x_batch, y_batch = x_train[r_idx, :], y_train[r_idx, :]  # current batch
        # Optimize the network
        loss=_train([x_batch,y_batch])
        print("= epoch : {} loss {} =".format(e,loss[0]))
        # plot result
        if e % 50 == 0:
            model.plot_result(x_test, e, _x_train=x_train, _y_train=y_train)
            model.plot_variances(x_test,e)

# Training data
x_min,x_max,n_train_half,y_max,var_scale = 0,100,1000,100,1.0 # 0,100,1000,100,0.5
x_train = np.linspace(x_min,x_max,n_train_half,dtype=np.float32).reshape((-1,1)) # [1000 x 1]
y_train = np.concatenate((y_max*np.sin(2.0*np.pi*x_train/(x_max-x_min))+2*y_max*x_train/x_max,
                          y_max*np.cos(2.0*np.pi*x_train/(x_max-x_min)))+2*y_max*x_train/x_max,
                          axis=1) # [1000 x 2] 1:2 대응관계

x_train,y_train = np.concatenate((x_train,x_train),axis=0),np.concatenate((y_train,-y_train),axis=0)
n_train = y_train.shape[0]
y_train = y_train + var_scale*y_max*np.random.randn(n_train,2)*np.square(1-x_train/x_max) # add noise
nzr_x_train = nzr(x_train)
x_train = nzr_x_train.get_nzdval(x_train) # normalize training input
y_train = nzr(y_train).nzd_data # normalize training output

# Train the mixture density network
max_iter = 20000
x_test = np.linspace(x_min,x_max,500, dtype=np.float32).reshape((-1,1))
x_test = nzr_x_train.get_nzdval(x_test) # normalize test input

if __name__ =='__main__':
    K.set_session(sess)
    M = MDN_reg_class(_x_dim=1, _y_dim=2, _k=20, _hids=[128, 128], _actv=opt.activation,
                      _sig_max=1.0, _SCHEDULE_SIG_MAX=True, _l2_reg_coef=1e-5,
                      _sess=sess, _VERBOSE=False)

    print("[%s] instantiated" % (M.name))

    M.compile(optimizer="rmsprop", loss="mse")
    if opt.optimizer =='rmsp':
        optimizer = RMSprop(opt.lr)
    else :
        optimizer = Adam(opt.lr)
    # plot_model(M, to_file='model.png')
    M(x_train[:opt.batch_size])
    # train
    train(M, optimizer, x_train, y_train, x_test, opt.num_epoch, opt.batch_size)