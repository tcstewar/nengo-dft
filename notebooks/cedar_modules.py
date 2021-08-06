import numpy as np
import scipy.signal
import nengo

##############################################################################
############################ helper functions ################################
class AbsSigmoid(object):
    ''' Creates an object that computes the AbsSigmoid function from cedar:
            0.5 * (1 + beta * (x-threshold) / 1 + beta * abs(x-threshold))
        A sigmoid function takes any real value as input and maps it to the 
        range between 0 and 1.

        Parameters
        ----------
        beta : int or float, optional
            Beta value of the AbsSigmoid function
        threshold : int or float, optional
            Threshold for which the AbsSigmoid should output 0.5
    '''
    def __init__(self, beta=100, threshold=0):
        self.beta = beta
        self.threshold = threshold

    def __call__(self, x):
        ''' Call method of the AbsSigmoid. Computes the AbsSigmoid output value
            for any given input value or values.

            Parameters
            ----------
            x : array_like, int or float

            Returns
            -------
            array_like, int or float
        '''
        return 0.5 * (1 + self.beta * (x-self.threshold) / (1 + self.beta * np.abs(x-self.threshold)))

def one_dimensional_peak(p, std, a, size=10):
    ''' Computes values of a 1-dimensional gaussian shaped peak.

        Parameters
        ----------
        p : int or float
            Position of the peak of the Gaussian
        std : int or float
            Standard deviation of the Gaussian
        a : int or float
            Amplitude of the peak
        size : int, optional
            Range and number of values of the peak

        Returns
        -------
        array_like
            values of the 1-dimensional peak
    '''
    x = np.arange(size)

    activations = a * np.exp(-0.5*(x-p)**2/std**2)

    return activations

def two_dimensional_peak(p, std, a, size=[10,10]):
    ''' Computes values of a 2-dimensional gaussian shaped peak.

        Parameters
        ----------
        p : list of ints or list of floats
            Position of the peak of the Gaussian
        std : list of ints or list of floats
            Standard deviation of the gaussian in both dimensions
        a : int or float
            Amplitude of the peak
        size : list of ints, optional
            Range and number of values of the peak

        Returns
        -------
        array_like
            Values of the 2-dimensional peak as 2-d array
    '''
    
    x = np.arange(size[1])
    y = np.arange(size[0])
    grid_x, grid_y = np.meshgrid(x, y)
    

    activations = a * np.exp(-0.5*((grid_x - p[1])**2/std[1]**2 + \
                                   (grid_y - p[0])**2/std[0]**2))

    return activations

def three_dimensional_peak(p, std, a, size=[10,10,10]):
    ''' Computes values of a 3-dimensional gaussian shaped peak.

        Parameters
        ----------
        p : list of ints or list of floats
            Position of the peak of the Gaussian
        std : list of ints or list of floats
            Standard deviation of the gaussian in all three dimensions
        a : int or float
            Amplitude of the peak
        size : list of ints, optional
            Range and number of values of the peak

        Returns
        -------
        array_like
            Values of the 3-dimensional peak as 3-d array
    '''
    x = np.arange(size[0])
    y = np.arange(size[1])
    z = np.arange(size[2])
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)

    activations = a * np.exp(-0.5 * ((grid_x - p[0])**2 /std[0]**2 + \
                                     (grid_y - p[1])**2 /std[1]**2 + \
                                     (grid_z - p[2])**2 /std[2]**2))

    return activations

def make_gaussian(sizes, centers, sigmas, a, normalize):
    ''' Computes a gaussian shaped peak.

        Parameters
        ----------
        sizes : list of ints
            List of the size (Range) of the gaussian per dimension
        centers : list of ints or list of floats
            Position of the peak of the gaussian
        sigmas : list of ints or list of floats
            Standard deviations per dimension of the gaussian
        a : int or float
            Amplitude of the gaussian

        Returns
        -------
        array_like 
            Values of the gaussian as array
    '''
    if len(sizes) == 1:
        activations = one_dimensional_peak(centers[0], sigmas[0], a, sizes[0])
        
    elif len(sizes) == 2:
        activations = two_dimensional_peak(centers, sigmas, a, sizes)
        
    elif len(sizes) == 3:
        activations = three_dimensional_peak(centers, sigmas, a, sizes)
        
    if normalize:
        activations = a * activations / np.sum(activations)
    # TODO: what to do if len(sizes) not between 1 and 3?
    return activations

def reduce(inp, dimension_mapping, compression_type):
    ''' Reduces an input along one or more dimensions by computing the 
        maximum or sum along these dimensions. 

        Parameters
        ----------
        inp : array_like
            input whose dimensions should be reduced
        dimension_mapping : dict
            Maps input dimensions to output dimensions. If a dimension is
            mapped to False it should be removed
        compression_type : {'max', 'sum'}
            Type of compression that should be used for the dimensions 
            that are removed

        Returns
        -------
        array_like
            Reduced input

        Notes
        -----
        Implemented for the Projection class.
    '''
    # get axis/axes to reduce
    axis_reduce = tuple([int(key) for key in dimension_mapping if type(dimension_mapping[key]) == bool])
    # reduce inp by using sum or max
    if compression_type == 'sum':
        out = np.sum(inp, axis=axis_reduce)
    elif compression_type == 'max':
        out = inp.max(axis=axis_reduce)
    
    # transpose array st order of dimensions is as given in dimension mapping
    dim_order = [int(value) for value in dimension_mapping.values() if type(value) == int]
    # print(dim_order, dimension_mapping)
    out = out.transpose(*dim_order)
    
    return out

def add_dimensions(inp, dimension_mapping, out_sizes):
    ''' Adds dimensions to a given input by repeating values
        along the new dimensions.

        Parameters
        ----------
        inp : array_like
            Input to which dimensions should be added
        dimension_mapping : dict
            Maps input to output dimensions
        out_sizes : list of ints
            Defines the size of each output dimension. Has to be the same
            as the size of the input dimensions for the output dimensions
            that these are mapped to.

        Returns
        -------
        array_like
            Input with added dimensions

        Notes
        -----
        Implemented for the Projection class
    '''
    
    added_dims = list(np.arange(len(out_sizes)))
    mapped_dims = [int(value) for value in dimension_mapping.values()]
    [added_dims.remove(dim) for dim in mapped_dims]

    # add dimensions
    for dim in added_dims:
        # add the dimension
        inp = np.expand_dims(inp, axis=dim)
        # repeat dimension to get right size
        inp = np.repeat(inp, repeats=out_sizes[dim], axis=dim)
        
    return inp

class GaussKernel(object):
    ''' Computes a kernel with a gaussian peak. 

        Parameters
        ----------
        c : int or float
            Multiplication factor for the kernel values
        sigma : int or float
            Standard deviation of the gaussian. Also specifies the
            size of the kernel
        normalize : Bool, optional
            Defines if kernel values should be normalized before
            multiplication with `c`
        dims : int, optional
            Dimensionality of the kernel. Should be between 0 and 3.

        Attributes
        ----------
        kernel_width : int
            Size/Width of the kernel
        kernel_matrix : array_like
            Kernel values
    '''
    def __init__(self, c, sigma, normalize=True, dims=2):
        self.c = c
        self.sigma = sigma
        self.normalize = normalize
        self.kernel_width = int(np.ceil(sigma*5)) # limit is always 5 
        # kernel width should always be an odd number --> like that in cedar
        if self.kernel_width % 2 == 0:
            self.kernel_width += 1
        self.dims = dims
        
        x = np.arange(self.kernel_width)
        cx = self.kernel_width//2
        
        if self.dims == 1 or self.dims == 0:
            dx = np.abs(x - cx)
            
        elif self.dims == 2:
            grid_x, grid_y = np.meshgrid(x, x)
            dx = np.sqrt((grid_x - cx)**2 + (grid_y - cx)**2)
         
        elif self.dims == 3:
            grid_x, grid_y, grid_z = np.meshgrid(x, x, x)
            dx = np.sqrt((grid_x - cx)**2 + (grid_y - cx)**2 + (grid_z - cx)**2)
            
        kernel_matrix = np.exp(-dx**2 / (2*self.sigma**2))
        if self.normalize:
            kernel_matrix /= np.sum(kernel_matrix)
        self.kernel_matrix = self.c * kernel_matrix
        
        # if the kernel is 0-dimensional it only consists of the central scalar value
        # need to compute from 1-dimensional kernel for normalizaton to work correctly
        if self.dims == 0:
            self.kernel_matrix = self.kernel_matrix[cx]
            
    def __call__(self):
        return self.kernel_matrix
    
class BoxKernel(object):
    ''' Computes a 0-dimensional BoxKernel, i.e. object that saves a 
        scalar value.  
    
        Parameters
        ----------
        amplitude : int or float
            Value of the scalar kernel

        Attributes
        ----------
        dims : 0
            Dimensionality of the kernel, only informational value
            to be consistent with cedar. Is always 0 since the BoxKernel
            for other dimensionalities is not implemented.
        kernel_matrix : int or float
            Same as `amplitude`. 

        Notes
        -----
        Implementation of the BoxKernel of cedar.
        Since the BoxKernel is only used with 0-dimensional fields in the 
        architecture this is a simplified implementation of the BoxKernel
        without the width parameter which is only needed for 1- or higher 
        dimensional fields.
    '''
    def __init__(self, amplitude):
        # dimensionality of BoxKernel always 0
        self.dims = 0 
        self.amplitude = amplitude
        self.kernel_matrix = amplitude
        
    def __call__(self):
        return self.kernel_matrix
    
def pad_and_convolve(inp, kernel, pad_width):
    ''' Applies cyclic padding to an input and then convolves it with a 
        given kernel. 

        Parameters
        ----------
        inp : array_like or float
            Input that should be padded and convolved with
        kernel : array_like or float
            Kernel to use for the convolution
        pad_width : tuple of ints
            Pad width on both sides of the input

        Returns
        -------
        array_like or float
            Result of the convolution

    '''
    # test if input consists of more than one number, i.e. field not 0-dimensional
    if inp.shape != ():
        # pad the input
        inp_padded = np.pad(inp, pad_width=pad_width, mode='wrap')

    # otherwise the kernel is just a scalar and can directly be convolved with the 
    # scalar input
    else:
        inp_padded = inp
    # perform the convolution
    conv = scipy.signal.convolve(inp_padded, kernel, mode='valid')
    
    return conv

def create_template(sizes, invert_sides, horizontal_pattern, sigma_th, 
                    mu_r, sigma_r, sigma_sigmoid):
    ''' Computes cedar's spatial template.

        Parameters
        ----------
        sizes : list of ints
            Size of the template per dimension
        invert_sides : Bool
            Defines if left and right should be inverted
        horizontal_pattern : Bool
            Defines if the pattern should be a horizontal or a vertical
            pattern
        sigma_th : float
        mu_r : float
        sigma_r : float
        sigma_sigmoid : float

        Returns
        -------
        array_like
            Spatial Pattern of left/right/above/below
    '''
    if invert_sides:
        invert_sides = -1
    else:
        invert_sides = 1
        
    size_x = sizes[0]
    size_y = sizes[1]
    
    shift_x = ((size_x - 1) / 2)
    shift_y = ((size_y - 1) / 2)
    
    x_grid, y_grid = np.meshgrid(np.arange(size_x), np.arange(size_y))
    
    x_shifted = x_grid - shift_x
    y_shifted = y_grid - shift_y
    
    x = x_shifted
    y = y_shifted 
    
    if horizontal_pattern:
        x = y_shifted
        y = x_shifted
        
    th = np.arctan2(y, invert_sides * x)
    r = np.log(np.sqrt(x**2 + y**2))
    
    gaussian = np.exp(-0.5 * th**2 / sigma_th**2 \
                      - 0.5 * (r - mu_r)**2 / sigma_r**2)
    sigmoid = invert_sides * AbsSigmoid()(x)
    
    pattern = (1 - sigma_sigmoid) * gaussian + sigma_sigmoid * sigmoid
    
    return pattern.transpose(1,0)
##############################################################################
############################# module classes #################################
class NeuralField(object):
    ''' Neural Field class similar to the Neural Field used by cedar.
        Implements the Neural Field equation of Dynamic Field theory.
        
        Paramters
        ---------
        sizes : list of ints
            Size per dimension
        h : int or float
            Resting level of the NeuralField
        tau : int or float
            Time scale parameter
        kernel : GaussKernel or BoxKernel instance or list of two of these
            Kernel instance (or instances) that is used for the convolution
        c_glob : int or float, optional
            Global inhibition parameter, determines the strength of global
            inhibition
        nonlinearity : callable, optional
            Sigmoid function callable that returns the sigmoid values
            when called on some input
        border_type : {'zero-filled borders', 'cyclic'}, optional
            Border type that should be used for the padding in the 
            convolution
        input_noise_gain : int or float, optional
            Sets the strength of the global noise
        name : NoneType or str, optional
            Name to use for the nengo.Node instance

        Attributes
        ----------
        u : array_like
            Activation of the NeuralField
        kernel_matrix : array_like or float
            Kernel values 
        node : nengo.Node instance
            Node instance that takes care of the actual computations
            of the NeuralField. Only created after calling method 
            `make_node()`
        
        Notes
        -----
        The dimensionality of the Neural Field is inferred from the number
        of elements in the sizes parameter. 
    '''
    def __init__(self, 
                 sizes, 
                 h, 
                 tau, 
                 kernel, 
                 c_glob=1, 
                 nonlinearity=AbsSigmoid(beta=100),
                 border_type='zero-filled borders',
                 input_noise_gain=0.1,
                 name=None):
        self.u = np.ones(sizes) * h
        self.h = h
        self.tau = tau
        self.sizes = sizes
        self.c_glob = c_glob
        self.nonlinearity = nonlinearity
        self.border_type = border_type
        self.input_noise_gain = input_noise_gain
        
        self.hist_a = []
        self.hist_c_glob = []
        self.hist_recurr = []
        
        # TODO: assertion not working anymore, since kernel can also be list now
        # assert (kernel.dims == len(sizes) or (len(sizes) == 0 and kernel.dims == 1)), \
        #         "Kernel must have same number of dimensions as Neural Field!"
        self.kernel = kernel
        
        # add kernels? other option would be to convolve with both kernels
        # in the update step
        # TODO: move this to outside NeuralField initialization
        if type(kernel) == list:
            km1 = kernel[0]()
            km2 = kernel[1]()
            if km1.shape == km2.shape:
                self.kernel_matrix = km1 + km2
            else:
                k_big = km1 if km1.shape[0] > km2.shape[0] else km2
                k_small = km1 if km1.shape[0] < km2.shape[0] else km2

                k_mask = np.zeros(k_big.shape)
                i_start = (k_big.shape[0]-k_small.shape[0])//2
                i_end = i_start + k_small.shape[0]
                if len(k_big.shape) == 1:
                    k_mask[i_start:i_end] = k_small
                elif len(k_big.shape) == 2:
                    k_mask[i_start:i_end, i_start:i_end] = k_small
                elif len(k_big.shape) == 3:
                    k_mask[i_start:i_end,i_start:i_end,i_start:i_end] = k_small
                self.kernel_matrix = k_big + k_mask
                       
        else:
            self.kernel_matrix = kernel()
        self.name = name
        
    
    def update_cyclic(self, stim, pad_width):
        ''' Update method for `border_type` cyclic

            Parameters
            ----------
            stim : array_like
                Input to the NeuralField
            pad_width : tuple of ints
                Pad width on both sides of the input

            Returns
            -------
            array_like
                Updated activation of the NeuralField after one step of 
                the neural field equation was applied
        '''
        a = self.nonlinearity(self.u)
        recurr = pad_and_convolve(a, self.kernel_matrix, pad_width)
        self.u += (-self.u + self.h + self.c_glob * np.sum(a) + recurr + stim)/self.tau + \
                  (self.input_noise_gain * np.random.randn(*self.sizes)) / self.tau 
        return self.u

    def update_zeros(self, stim):
        ''' Update method for `border_type` zero-filled borders

            Parameters
            ----------
            stim : array_like
                Input to the NeuralField
            
            Returns
            -------
            array_like
                Updated activation of the NeuralField after one step of 
                the neural field equation was applied
        '''
        a = self.nonlinearity(self.u)
        recurr = scipy.signal.convolve(a, self.kernel_matrix, mode='same')
        self.u += (-self.u + self.h + self.c_glob * np.sum(a) + recurr + stim)/self.tau + \
                  (self.input_noise_gain * np.random.randn(*self.sizes)) / self.tau 
        return self.u


    def make_node(self):
        ''' Creates a nengo.Node() instance that takes care of the 
            NeuralField computations. It uses one of the above update
            methods for its update in each time step. 
        '''
        if self.border_type == 'zero-filled borders':
            # call convolve only update
            update = lambda t, x: self.update_zeros(x.reshape(self.sizes)).flatten()
        else:
            # bordertype is cyclic, need pad_and_convolve update
            # the 1 that is used as the kernel_width for 0-dimensional kernels is not really used
            # in later computations but needs to be specified here, otherwise, the shape[0] would
            # throw an error
            kernel_width = self.kernel_matrix.shape[0] if self.kernel_matrix.shape != () else 1
            pad_width_f = kernel_width//2
            pad_width_b = kernel_width//2 if kernel_width % 2 == 1 else kernel_width//2 - 1 
            pad_width = (pad_width_f, pad_width_b)

            update = lambda t, x: self.update_cyclic(x.reshape(self.sizes),
                                                     pad_width).flatten()

        if self.name is not None:
            self.node = nengo.Node(update,
                          size_in=int(np.product(self.sizes)), 
                          size_out=int(np.product(self.sizes)),
                          label=self.name)
        else:
            self.node = nengo.Node(update,
                          size_in=int(np.product(self.sizes)), 
                          size_out=int(np.product(self.sizes)))
                          

class ComponentMultiply(object):
    ''' Class around nengo.Node instance that computes the elementwise
        multiplication of two inputs similar to cedar's ComponentMultiply

        Parameters
        ----------
        inp_size1 : list of ints
            Size per dimension of the first input
        inp_size2 : list of ints
            Size per dimension of the second input
        name : None or str, optional
            Name to use for the nengo.Node instance

        Attributes
        ----------
        out_size : list of ints
            Size per dimension of the output
        connections : int
            Number of connections made to the ComponentMultiply instance
        node : nengo.Node instance
            Node instance that takes care of the actual computations
            of the ComponentMultiply. Only created after calling method 
            `make_node()`

        Notes
        -----
        The two inputs can have different sizes and then the multiplication 
        is not an elementwise multiplication any more, but it does 
        correspond to the standard multiplication of numpy arrays.
    '''
    def __init__(self, inp_size1, inp_size2, name=None):
        self.inp_size1 = inp_size1
        self.inp_size2 = inp_size2
        if len(self.inp_size1) >= len(self.inp_size2):
            self.out_size = self.inp_size1
        else:
            self.out_size = self.inp_size2
        # internal counter for number of connections to know where to connect to
        self.connections = 0
        self.name = name
        
    def update(self, inp):
        ''' Update method that the nengo.Node uses

            Parameters
            ----------
            inp : array_like
                Inputs to the ComponentMultiply node

            Returns
            -------
            array_like
                Elementwise multiplication of the two inputs
        '''
        # get the index of where input1 and input2 are seperated
        sep = int(np.prod(self.inp_size1))
        inp1 = inp[:sep]
        inp2 = inp[sep:]
        
        # if the shape is the same reshaping the inputs is not necessary
        if self.inp_size1 == self.inp_size2 or self.inp_size1 == [] or self.inp_size2 == []:
            return inp1 * inp2
        
        # if the shapes are not the same the inputs have to be reshaped
        # to their real unflattened form before multipliying
        else:
            inp1 = inp1.reshape(*self.inp_size1)
            inp2 = inp2.reshape(*self.inp_size2)
            return (inp1 * inp2).flatten()
    
    def make_node(self):
        ''' Creates a nengo.Node() instance that takes care of the 
            ComponentMultiply computations. It uses the above update
            method for its update in each time step. 
        ''' 
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x), 
                          size_in=int(np.prod(self.inp_size1)+np.prod(self.inp_size2)), 
                          size_out=int(np.prod(self.out_size)),
                          label=self.name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x), 
                          size_in=int(np.prod(self.inp_size1)+np.prod(self.inp_size2)), 
                          size_out=int(np.prod(self.out_size)))


class GaussInput(object):
    ''' Class around nengo.Node instance that creates a gaussian shaped
        input for a cedar architecture, similar to cedar's GaussInput.

        Parameters
        ----------
        sizes : list of ints
            Size of GaussInput per dimension
        centers : list of ints or list of floats
            Position of the gaussian peak
        sigmas : list of ints or list of floats
            Standard deviation of the gaussian per dimension
        a : int or float
            Amplitude of the gaussian
        name : NoneType or str, optional
            Name to use for the nengo.Node instance

        Attributes
        ----------
        node : nengo.Node instance
            Node instance that outputs the gaussian input. Only created
            after calling method `make_node()`
    '''
    def __init__(self, sizes, centers, sigmas, a, normalize, name=None):
        
        # add asserts to check if sizes same length as centers and sigmas
        self.sizes = sizes
        self.centers = centers
        self.sigmas = sigmas
        self.a = a
        self.name = name
        self.normalize = normalize
        
    def make_node(self):
        ''' Creates a nengo.Node() instance that has the gaussian shaped
            input as a constant output. 
        ''' 
        if self.name is not None:
            self.node = nengo.Node(make_gaussian(self.sizes, self.centers, self.sigmas, self.a, self.normalize).flatten(),
                                   label=self.name)
        else:
            self.node = nengo.Node(make_gaussian(self.sizes, self.centers, self.sigmas, self.a, self.normalize).flatten())
        return self.node


class ConstMatrix(object):
    ''' Class around nengo.Node instance that creates a constant matrix as
        an input for a cedar architecture, similar to cedar's ConstMatrix.

        Parameters
        ----------
        sizes : list of ints
            Size of the constant matrix per dimension
        value : float or int
            Constant value that is used for the constant matrix
        name : NoneType or str, optional
            Name to use for the nengo.Node instance

        Attributes
        ----------
        node : nengo.Node instance
            Node instance that outputs the constant matrix. Only created
            after calling method `make_node()`
    '''
    def __init__(self, sizes, value, name=None):
        self.sizes = sizes
        self.value = value
        self.name = name
        
    def make_node(self):
        ''' Creates a nengo.Node() instance that has the constant matrix
            as an output. 
        ''' 
        if self.name is not None:
            self.node = nengo.Node(np.ones(int(np.prod(self.sizes)))*self.value,
                                   label=self.name)
        else:
            self.node = nengo.Node(np.ones(int(np.prod(self.sizes)))*self.value)


class StaticGain(object):
    ''' Class around nengo.Node instance that multiplies its input with
        a constant value.

        Parameters
        ----------
        sizes : list of ints
            Size of the input and output per dimension
        gain_factor : float or int
            Value to multiply the input with
        name : NoneType or str, optional
            Name to use for the nengo.Node instance

        Attributes
        ----------
        node : nengo.Node instance
            Node instance that performs the static gain multiplication. 
            Only created after calling method `make_node()`

    '''
    def __init__(self, sizes, gain_factor, name=None):
        self.sizes = sizes
        self.gain_factor = gain_factor
        self.name = name
        
    def update(self, inp):
        ''' Update method for the nengo.Node 

            Parameters
            ---------
            inp : array_like
                Input to the nengo.Node

            Returns
            -------
            array_like
                The input multiplied with `gain_factor`

        '''
        return inp * self.gain_factor
    
    def make_node(self):
        ''' Creates a nengo.Node() instance that takes care of the 
            StaticGain computations. It uses the above update
            method for its update in each time step. 
        ''' 
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=int(np.prod(self.sizes)),
                                   label=self.name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=int(np.prod(self.sizes)))


class Flip(object):
    ''' Class around nengo.Node that flips a 2-dimensional input along
        the first or the second dimension or both.

        Parameters
        ----------
        sizes : list of two ints
            Size of first and second dimension of the input
        flip_dimensions : list of two bools
            Defines whether the first or second dimension or both should
            be flipped
        name : NoneType or str, optional
            Name to use for the nengo.Node instance

        Attributes
        ----------
        node : nengo.Node instance
            Node instance that performs the flip computations. Only 
            created after calling method `make_node()`
    '''
    def __init__(self, sizes, flip_dimensions, name=None):
        self.sizes = sizes
        self.flip_dimensions = flip_dimensions
        self.name = name
        
    def update(self, inp):
        ''' Update method for the nengo.Node. Flips the input along the
            dimensions defined by `flip_dimensions`.

            Parameters
            ----------
            inp: array_like
                Input to perform the flips on

            Returns
            -------
            array_like
                The flipped input
        '''
        out = inp.reshape(*self.sizes)
        # flip along first dimension?
        if self.flip_dimensions[0]:
            out = np.flip(out, axis=0)
        # flip along second dimension?
        if self.flip_dimensions[1]:
            out = np.flip(out, axis=1)
            
        return out.flatten()
    
    def make_node(self):
        ''' Creates a nengo.Node() instance that takes care of the 
            Flip computations. It uses the above update
            method for its update in each time step. 
        ''' 
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=int(np.prod(self.sizes)), 
                          size_out=int(np.prod(self.sizes)), label=self.name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x), size_in=int(np.prod(self.sizes)), 
                          size_out=int(np.prod(self.sizes)))


class Projection(object):
    ''' Class around a nengo.Node instance that projects and input of one
        dimensionality to another dimensionality.

        The input can either be projected to a higher dimensionality by
        repeating the value along a dimension or it can be projected to a
        lower dimensionality by computing the sum or taking the maximum
        along a dimension.

        Parameters
        ----------
        sizes_out : list of ints
            Output size of the nengo.Node per dimension
        sizes_in : list of ints
            Input size of the nengo.Node per dimension
        dimension_mapping : dict
            Dictionary that defines where each input dimension is mapped to.
            The values can be either the index of one of the output 
            dimensions or False if the input dimension should be removed
        compression_type : {'max', 'sum'}, optional
            Type of compression to use when removing a dimension. The 
            value is ignored when performing an upscaling of dimensions
        name : NoneType or str, optional
            Name to use for the nengo.Node instance

        Attributes
        ----------
        node : nengo.Node instance
            Node instance that performs the projection. Only created
            after calling method `make_node()`
    '''
    def __init__(self, sizes_out, sizes_in, dimension_mapping, 
                 compression_type='max', name=None):
        self.sizes_out = sizes_out
        self.sizes_in = sizes_in
        self.dimension_mapping = dimension_mapping
        self.compression_type = compression_type
        self.name = name
        self.index_maps = None
        
    def update(self, inp):
        ''' Update method for the nengo.Node update. Performs the 
            projection computations for a given input.

            Parameters
            ----------
            inp : array_like
                Input of the projection

            Returns
            -------
            array_like
                The projected output
        '''
        # reshape inp
        if self.sizes_in != []:
            out = inp.reshape(*self.sizes_in)
        else:
            out = inp
        
        # either downsizing
        if len(self.dimension_mapping) > len(self.sizes_out):
            out = reduce(out, self.dimension_mapping, self.compression_type)
            
        # or upsizing
        elif len(self.sizes_out) > len(self.dimension_mapping):
            out = add_dimensions(out, self.dimension_mapping, self.sizes_out)
            
        if out.shape != self.sizes_out:
            if self.index_maps is None:
                self.index_maps = self.create_index_maps(out.shape, self.sizes_out)
            
            for i, m in enumerate(self.index_maps):
                out = out[(slice(None, None, None),)*i + (m,)]
            
        return out.flatten()
        
        
    def create_index_maps(self, size_ins, size_outs):
        maps = []
        for i in range(len(size_ins)):
            maps.append(np.linspace(0, size_ins[i]-1, size_outs[i]).astype(int))
        return maps    
        
               
    def make_node(self):
        ''' Creates a nengo.Node() instance that takes care of the 
            Projection computations. It uses the above update method for
            its update in each time step. 
        ''' 
        if self.name is not None:
            self.node = nengo.Node(lambda t, x: self.update(x), 
                          size_in=int(np.prod(self.sizes_in)) if self.sizes_in != [] else 1, 
                          size_out=int(np.prod(self.sizes_out)) if self.sizes_out != [] else 1,
                          label=self.name)
        else:
            self.node = nengo.Node(lambda t, x: self.update(x), 
                          size_in=int(np.prod(self.sizes_in)) if self.sizes_in != [] else 1, 
                          size_out=int(np.prod(self.sizes_out)) if self.sizes_out != [] else 1)
        return self.node


class Convolution(nengo.Network):
    def __init__(self, amplitude, sigma, limit, input_shape, label=None):
        super().__init__(label=label)
        
        assert len(input_shape) in [1,2]
        
        max_width = np.max(input_shape)
        x = np.arange(0, max_width)
        k = np.exp(-0.5*((x)/sigma)**2)
        width = int(np.ceil(sigma*limit))
        k = k[:-width]
        
        if len(input_shape)==2:
            xx, yy = np.meshgrid(np.arange(len(k)), np.arange(len(k)))
            dist = xx**2 + yy**2
            k = np.exp(-0.5*(dist/sigma**2))
        
        
        k = np.concatenate([k[1:][::-1], k])
        
        if len(input_shape)==2:
            k = np.hstack([k[:,1:][:,::-1], k])
        
        k = k * amplitude / np.sum(k)
    
        with self:
            n_neurons = np.prod(input_shape)
            strides = np.ones_like(input_shape)
            input_shape = tuple(input_shape + [1])
            
            self.input = nengo.Node(None, size_in=n_neurons)
            self.output = nengo.Node(None, size_in=n_neurons)
            
            t = nengo.Convolution(n_filters=1, input_shape=input_shape, kernel_size=k.shape, strides=strides, 
                                  padding='same', init=k[...,None,None])
            nengo.Connection(self.input, self.output, transform=t, synapse=None)
        
    
class SpatialTemplate(object):
    ''' Class around nengo.Node that creates a spatial template as an
        input of a cedar architecture.

        Parameters
        ----------
        sizes : list of ints
            Size of the spatial template per dimension
        invert_sides : Bool
            Defines if left and right should be inverted
        horizontal_pattern : Bool
            Defines if the pattern should be a horizontal or a vertical
            pattern
        sigma_th : float
        mu_r : float
        sigma_r : float
        sigma_sigmoid : float
        name : NoneType or str, optional
            Name to use for the nengo.Node instance

        Attributes
        ----------
        node : nengo.Node instance
            Node instance that outputs the spatial pattern. Only created
            after calling method `make_node()`

        Notes
        -----
        The SpatialTemplate is used for the creation of a left, right, 
        above or below activation pattern.
    '''
    def __init__(self, sizes, invert_sides, horizontal_pattern, sigma_th_hor, 
                 mu_r, sigma_r, sigma_sigmoid_fw, name=None):
        self.sizes = sizes
        self.invert_sides = invert_sides
        self.horizontal_pattern = horizontal_pattern
        self.sigma_th_hor = sigma_th_hor
        self.mu_r = mu_r
        self.sigma_r = sigma_r
        self.sigma_sigmoid_fw = sigma_sigmoid_fw
        self.name = name
        
    def make_node(self):
        ''' Creates a nengo.Node() instance that outputs the spatial 
            pattern. It uses the above update method for its update in 
            each time step.
        ''' 
        if self.name is not None:
            self.node = nengo.Node(create_template(self.sizes, self.invert_sides, 
                                          self.horizontal_pattern, 
                                          self.sigma_th_hor, self.mu_r, 
                                          self.sigma_r, 
                                          self.sigma_sigmoid_fw).flatten(), 
                                   label=self.name)
        else:
            self.node = nengo.Node(create_template(self.sizes, self.invert_sides, 
                                          self.horizontal_pattern,
                                          self.sigma_th_hor, self.mu_r, 
                                          self.sigma_r, 
                                          self.sigma_sigmoid_fw).flatten())


class Boost(object):
    ''' Class around nengo.Node that sends a signal of a certain strength
        if active and otherwise just sends 0.

        Parameters
        ----------
        strength : int or float
            Signal strength of active node
        name : NoneType or str, optional
            Name to use for the nengo.Node instance

        Attributes
        ----------
        active : bool
            Defines whether the nodes is active or not
        node : nengo.Node instance
            Node instance that sends the signal. Only created after 
            calling method `make_node()`
    '''
    def __init__(self, strength, name=None):
        self.strength = strength
        self.name = name
        self.active = False
    
    def update(self):
        ''' Update method for the nengo.Node. Depending on whether the 
            node is active or not sends a signal of strenght `strength`
            or 0.

            Returns
            -------
            float or int
                The signal
        '''
        # Check if the node is active or not
        if self.active:
            return self.strength
        else:
            return 0

    def make_node(self):
        ''' Creates a nengo.Node() instance that sends the signal. It uses
            the above update method for its update in each time step. 
        ''' 
        if self.name is not None:
            # the strength value should only be the node's value when it 
            # it is active, otherwise the default value of the node is 0
            self.node = nengo.Node(lambda t: self.update(), label=self.name)
        else:
            self.node = nengo.Node(lambda t: self.update())