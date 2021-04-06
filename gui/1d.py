import nengo
import numpy as np
import nengo_dft

N = 100

model = nengo.Network()
with model:
    dft1a = nengo_dft.DFT(shape=[N], tau=0.020)
    dft1a.add_kernel(exc=17.5, inh=15)
    display1a = dft1a.make_display_1d()
    
    dft1b = nengo_dft.DFT(shape=[N], global_inh=0.9)
    dft1b.add_kernel(exc=21, inh=0)
    display1b = dft1b.make_display_1d()

    dft1c = nengo_dft.DFT(shape=[N])
    dft1c.add_kernel(exc=30, inh=27.5)    
    display1c = dft1c.make_display_1d()
    
    
    h = nengo.Node(-4)
    nengo.Connection(h, dft1a.h, synapse=None)
    nengo.Connection(h, dft1b.h, synapse=None)
    nengo.Connection(h, dft1c.h, synapse=None)
    
    x = np.arange(N)
    def convert_stim(p):
        width, pos, amp = p
        return amp*np.exp(-0.5*((x-pos)/width)**2)
        
    stim1 = nengo.Node([5, 25, 5.8])
    stim2 = nengo.Node([5, 50, 0])
    stim3 = nengo.Node([5, 75, 6])
    
    nengo.Connection(stim1, dft1a.s, synapse=None, function=convert_stim)
    nengo.Connection(stim2, dft1a.s, synapse=None, function=convert_stim)
    nengo.Connection(stim3, dft1a.s, synapse=None, function=convert_stim)
    nengo.Connection(stim1, dft1b.s, synapse=None, function=convert_stim)
    nengo.Connection(stim2, dft1b.s, synapse=None, function=convert_stim)
    nengo.Connection(stim3, dft1b.s, synapse=None, function=convert_stim)
    nengo.Connection(stim1, dft1c.s, synapse=None, function=convert_stim)
    nengo.Connection(stim2, dft1c.s, synapse=None, function=convert_stim)
    nengo.Connection(stim3, dft1c.s, synapse=None, function=convert_stim)
    

    
