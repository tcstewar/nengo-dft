import nengo
import numpy as np
import nengo_dft

N = 100

model = nengo.Network()
with model:
    dft2a = nengo_dft.DFT(shape=[N,N], tau=0.020)
    dft2a.add_kernel(exc=17.5, inh=15, epsilon=0.01)

    h = nengo.Node(-4)
    nengo.Connection(h, dft2a.h, synapse=None)

    x = np.arange(N)
    xx, yy = np.meshgrid(x, x)
    def convert_stim(p):
        width, pos_x, pos_y, amp = p
        return amp*np.exp(-0.5*(((xx-pos_x)**2+(yy-pos_y)**2)/width**2)).flatten()
        
    stim1 = nengo.Node([5, 25, 25, 5.8])
    stim2 = nengo.Node([5, 50, 50, 0])
    stim3 = nengo.Node([5, 75, 75, 6])
    
    nengo.Connection(stim1, dft2a.s, synapse=None, function=convert_stim)
    nengo.Connection(stim2, dft2a.s, synapse=None, function=convert_stim)
    nengo.Connection(stim3, dft2a.s, synapse=None, function=convert_stim)


    
