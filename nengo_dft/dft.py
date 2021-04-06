import nengo
import numpy as np

class DFT(nengo.Network):
    def __init__(self, shape, tau=0.02, c_noise=1, beta=4, global_inh=0, h=None):
        super().__init__()
        self.shape = shape
        with self:
            n_neurons = np.prod(shape)
            
            self.u = nengo.Node(None, size_in=n_neurons)
            self.g = nengo.Ensemble(n_neurons=n_neurons, dimensions=1,
                                    gain = np.ones(n_neurons)*beta,
                                    bias = np.zeros(n_neurons),
                                    neuron_type=nengo.Sigmoid(tau_ref=1),
                                   )
            self.s = nengo.Node(None, size_in=n_neurons)
            
            if h is None:
                self.h = nengo.Node(None, size_in=1)
            else:
                self.h = nengo.Node(h)
            
            nengo.Connection(self.u, self.g.neurons, synapse=None)
            
            self.noise = nengo.Node(nengo.processes.WhiteNoise(scale=False), size_out=n_neurons)
            
            
            
            self.du = nengo.Node(None, size_in=n_neurons)
            nengo.Connection(self.u, self.du, transform=-1, synapse=0)
            nengo.Connection(self.s, self.du, synapse=None)
            nengo.Connection(self.h, self.du, transform=np.ones((n_neurons, 1)), synapse=None)
            nengo.Connection(self.noise, self.du, transform=c_noise)
            
            # TODO: is there a better way to do this with a synapse so we don't hard-code dt?
            dt = 0.001
            nengo.Connection(self.u, self.u, synapse=0)
            nengo.Connection(self.du, self.u, synapse=None, transform=dt/tau)
            
            # TODO: is this more accurate?
            #nengo.Connection(self.du, self.u, synapse=None, transform=-(1-np.exp(dt/tau)))
            
            if global_inh != 0:
                self.global_inh = nengo.Node(None, size_in=1)
                nengo.Connection(self.g.neurons, self.global_inh, transform=-global_inh*np.ones((1, n_neurons)), synapse=None)
                nengo.Connection(self.global_inh, self.du, transform=np.ones((n_neurons, 1)), synapse=0)
                            
    def add_kernel(self, exc, inh, exc_width=5, inh_width=10, epsilon=0.001):
        assert len(self.shape) in [1,2]
        
        max_width = np.max(self.shape)
        x = np.arange(0, max_width)
        k_exc = np.exp(-0.5*((x)/exc_width)**2)
        k_inh = np.exp(-0.5*((x)/inh_width)**2)
        width = np.min([np.searchsorted(k_exc[::-1], epsilon), np.searchsorted(k_inh[::-1], epsilon)])
        k_exc = k_exc[:-width]
        k_inh = k_inh[:-width]
        
        if len(self.shape)==2:
            xx, yy = np.meshgrid(np.arange(len(k_exc)), np.arange(len(k_exc)))
            dist = xx**2 + yy**2
            k_exc = np.exp(-0.5*(dist/exc_width**2))
            k_inh = np.exp(-0.5*(dist/inh_width**2))            
        
        
        k_exc = np.concatenate([k_exc[1:][::-1], k_exc])
        k_inh = np.concatenate([k_inh[1:][::-1], k_inh])
        
        if len(self.shape)==2:
            k_exc = np.hstack([k_exc[:,1:][:,::-1], k_exc])
            k_inh = np.hstack([k_inh[:,1:][:,::-1], k_inh])
        
        k_exc = k_exc * exc / np.sum(k_exc)
        k_inh = k_inh * inh / np.sum(k_inh)
    
        k = k_exc - k_inh
        
        with self:
            n_neurons = np.prod(self.shape)
            input_shape = tuple(self.shape + [1])
            strides = np.ones_like(self.shape)
            t = nengo.Convolution(n_filters=1, input_shape=input_shape, kernel_size=k.shape, strides=strides, 
                                  padding='same', init=k[...,None,None])
            nengo.Connection(self.g.neurons, self.du, transform=t, synapse=0)
            
        return k


    def make_display_1d(self):

        template = '''
            <svg width="100%%" height="100%%" viewbox="0 0 100 100">
                
                <line x1=50 y1=0 x2=50 y2=100 stroke="#aaaaaa"/>
                <line x1=0 y1=50 x2=100 y2=50 stroke="#aaaaaa"/>
                %s
            </svg>'''
            
        def make_path(xpts, ypts, color):
            path = []
            for i, x in enumerate(xpts):
                path.append('%1.0f %1.0f' % (x, ypts[i]))
            p = 'L'.join(path)
            return '<path d="M%s" fill="none" stroke="%s"/>' % (p, color)

        N = np.prod(self.shape)
        def update(t, x):
            sh = x[:N]
            u = x[N:N*2]
            g = x[N*2:]
            
            import scipy.interpolate
            ymap = scipy.interpolate.interp1d((-15,15),(100,0), fill_value='extrapolate')
            
            xx = np.linspace(0, 100, N)
            paths = [
                make_path(xx, ymap(sh), color='green'),
                make_path(xx, ymap(u), color='blue'),
                make_path(xx, ymap(g*10), color='red')
                ]
                
            update._nengo_html_ = template % ''.join(paths)
            
        n = nengo.Node(update, size_in=N*3)
        nengo.Connection(self.s, n[:N], synapse=None)
        nengo.Connection(self.h, n[:N], synapse=None, transform=np.ones((N, 1)))
        nengo.Connection(self.u, n[N:N*2], synapse=None)
        nengo.Connection(self.g.neurons, n[N*2:], synapse=None)
        return n
