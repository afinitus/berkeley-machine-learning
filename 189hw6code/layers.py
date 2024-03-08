"""
Author: Sophia Sanborn, Sagnik Bhattacharya
Institution: UC Berkeley
Date: Spring 2020
Course: CS189/289A
Website: github.com/sophiaas, github.com/sagnibak
"""

import numpy as np
from abc import ABC, abstractmethod

from neural_networks.activations import initialize_activation
from neural_networks.weights import initialize_weights
from collections import OrderedDict
from neural_networks.utils.convolution import im2col, col2im
from typing import Callable, List, Literal, Tuple, Union


class Layer(ABC):
    """Abstract class defining the `Layer` interface."""

    def __init__(self):
        self.activation = None

        self.n_in = None
        self.n_out = None

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

        super().__init__()

    @abstractmethod
    def forward(self, z: np.ndarray) -> np.ndarray:
        pass

    def clear_gradients(self) -> None:
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict(
            {a: np.zeros_like(b) for a, b in self.gradients.items()}
        )

    def forward_with_param(
        self, param_name: str, X: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val: np.ndarray) -> np.ndarray:
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self) -> List[np.ndarray]:
        return [b for a, b in self.parameters.items()]

    def _get_cache(self) -> List[np.ndarray]:
        return [b for a, b in self.cache.items()]

    def _get_gradients(self) -> List[np.ndarray]:
        return [b for a, b in self.gradients.items()]


def initialize_layer(
    name: str,
    activation: str = None,
    weight_init: str = None,
    n_out: int = None,
    kernel_shape: Tuple[int, int] = None,
    stride: int = None,
    pad: int = None,
    mode: str = None,
    keep_dim: str = "first",
) -> Layer:
    """Factory function for layers."""
    if name == "fully_connected":
        return FullyConnected(
            n_out=n_out, activation=activation, weight_init=weight_init,
        )

    elif name == "conv2d":
        return Conv2D(
            n_out=n_out,
            activation=activation,
            kernel_shape=kernel_shape,
            stride=stride,
            pad=pad,
            weight_init=weight_init,
        )

    elif name == "pool2d":
        return Pool2D(kernel_shape=kernel_shape, mode=mode, stride=stride, pad=pad)

    elif name == "flatten":
        return Flatten(keep_dim=keep_dim)

    else:
        raise NotImplementedError("Layer type {} is not implemented".format(name))


class FullyConnected(Layer):
    """A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(
        self, n_out: int, activation: str, weight_init="xavier_uniform"
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.activation = initialize_activation(activation)

        # instantiate the weight initializer
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int]) -> None:
        """Initialize all layer parameters (weights, biases)."""
        self.n_in = X_shape[1]

        #we want the weights to be based off our n values
        #our b should all be 0 and set using out n out val

        W = self.init_weights((self.n_in, self.n_out))
        b = np.zeros((1, self.n_out))
        #for our cache and gradients we want it set to 0 and empty list
        #this way it is editable and we add over time
        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": [], "X": []})
        self.gradients = OrderedDict({"W": np.zeros_like(W),"b": np.zeros_like(b)})

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim)
        """
        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)

        #we want to set W and b according to our parameter predef
        #after that to get Z we want to use the weights and b vector
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = X @ W + b
        # perform an affine transformation and activation
        out = self.activation(Z)
        
        # store information necessary for backprop in `self.cache`
        self.cache["Z"] = Z
        self.cache["X"] = X

        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  derivative of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        #as usual we set W and b according to our parameter predef
        W = self.parameters["W"]
        b = self.parameters["b"]
        # unpack the cache
        Z = self.cache["Z"]
        X = self.cache["X"]
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        #we need to get dZ based off the value opf dLdY as found earlier
        #then we will use dZ to get dX and dW and dB since they are all
        #based off of it, and we use our regular formulas to get the values
        dZ = self.activation.backward(Z, dLdY)
        dX = dZ @ W.T
        dW = X.T @ dZ
        dB = dZ.sum(axis=0, keepdims=True)

        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        self.gradients["W"] = dW
        self.gradients["b"] = dB
        ### END YOUR CODE ###

        return dX


class Conv2D(Layer):
    """Convolutional layer for inputs with 2 spatial dimensions."""

    def __init__(
        self,
        n_out: int,
        kernel_shape: Tuple[int, int],
        activation: str,
        stride: int = 1,
        pad: str = "same",
        weight_init: str = "xavier_uniform",
    ) -> None:

        super().__init__()
        self.n_in = None
        self.n_out = n_out
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.pad = pad

        self.activation = initialize_activation(activation)
        self.init_weights = initialize_weights(weight_init, activation=activation)

    def _init_parameters(self, X_shape: Tuple[int, int, int, int]) -> None:
        """Initialize all layer parameters and determine padding."""
        self.n_in = X_shape[3]

        W_shape = self.kernel_shape + (self.n_in,) + (self.n_out,)
        W = self.init_weights(W_shape)
        b = np.zeros((1, self.n_out))

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache = OrderedDict({"Z": [], "X": []})
        self.gradients = OrderedDict({"W": np.zeros_like(W), "b": np.zeros_like(b)})

        if self.pad == "same":
            self.pad = ((W_shape[0] - 1) // 2, (W_shape[1] - 1) // 2)
        elif self.pad == "valid":
            self.pad = (0, 0)
        elif isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass for convolutional layer. This layer convolves the input
        `X` with a filter of weights, adds a bias term, and applies an activation
        function to compute the output. This layer also supports padding and
        integer strides. Intermediates necessary for the backward pass are stored
        in the cache.

        Parameters
        ----------
        X  input with shape (batch_size, in_rows, in_cols, in_channels)

        Returns
        -------
        output feature maps with shape (batch_size, out_rows, out_cols, out_channels)
        """
        if self.n_in is None:
            self._init_parameters(X.shape)

        W = self.parameters["W"]
        b = self.parameters["b"]

        #here we get the initial params of the function using the shapes
        kernel_height, kernel_width, in_channels, out_channels = W.shape
        n_examples, in_rows, in_cols, in_channels = X.shape
        kernel_shape = (kernel_height, kernel_width)

        ### BEGIN YOUR CODE ###
        #here we want to get the rows and columns going out the amtrix based off our padding and the in
        rout = int((2*self.pad[0] + in_rows - kernel_height) / self.stride + 1)
        cout = int((2*self.pad[1] + in_cols - kernel_width) / self.stride + 1)
        #we can manually implement the padding for X as follows which then forces us to use the for loops
        paddedX = np.pad(X, pad_width=((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode='constant')
        # implement a convolutional forward pass
        #we start with the empty size 4 matrix
        Z = np.empty((n_examples, rout, cout, out_channels), dtype=X.dtype)
        #now we implement the for loops and adjust Z according to the row column and channel
        for row in range(rout):
            for col in range(cout):
                for chan in range(out_channels):
                    #here we do the sum according to the formula which is complicated:
                    Z[:, row, col, chan] = (np.sum(paddedX[:,row * self.stride
                                                            : row * self.stride + kernel_height, col
                                                              * self.stride : col * self.stride + kernel_width,:,] 
                                                              * W[:, :, :, chan], axis=(1, 2, 3),)+ b[:, chan])
        #finalyl once we have our Z we want to run activation on it to create the proper Z value
        out = self.activation(Z)

        # cache any values required for backprop
        self.cache["Z"] = Z
        self.cache["X"] = X
        ### END YOUR CODE ###

        return out

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for conv layer. Computes the gradients of the output
        with respect to the input feature maps as well as the filter weights and
        biases.

        Parameters
        ----------
        dLdY  derivative of loss with respect to output of this layer
              shape (batch_size, out_rows, out_cols, out_channels)

        Returns
        -------
        derivative of the loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, in_channels)
        """
        ### BEGIN YOUR CODE ###
        #first we want to get all of our values from our cached parameters that we have saved
        W = self.parameters["W"]
        b = self.parameters["b"]
        Z = self.cache["Z"]
        X = self.cache["X"]
        #similar to what we did on our forward we use the W and X shapes to get the needed params
        kernel_height, kernel_width, in_channels, out_channels = W.shape
        # perform a backward pass
        n_examples, in_rows, in_cols, in_channels = X.shape
        #here just like in the forward code we get rout and cout
        rout = int((2*self.pad[0] + in_rows - kernel_height) / self.stride + 1)
        cout = int((2*self.pad[1] + in_cols - kernel_width) / self.stride + 1)
        #the difference is we need the gradients for the backwards to get the change and directionality
        dZ = self.activation.backward(Z, dLdY)
        #same padding format
        paddingX = np.pad(X, pad_width=((0, 0), (self.pad[0], self.pad[0]), (self.pad[1], self.pad[1]), (0, 0)), mode='constant')
        paddingdX = np.zeros_like(paddingX)
        #here we alos need dW and DB for the updates
        dW = np.zeros_like(W)
        dB = dZ.sum(axis=(0, 1, 2)).reshape(1, -1)
        #we run the for loops for our code the same way we ran them for our forward code
        for row in range(rout):
            for col in range(cout):
                for chan in range(out_channels):
                    #the only difference is now we change the pd dX and dW for the backwards descent
                    paddingdX[:,row * self.stride:row * 
                              self.stride + kernel_height,col * 
                              self.stride:col*self.stride + kernel_width,:] += W[np.newaxis, :, 
                                                                                 :, :, chan] * dZ[:, row:row + 1, 
                                                                                                  col:col + 1, np.newaxis, chan]
                    dW[:, :, :, chan] += np.sum(paddingX[:, row * self.stride:row * 
                                                         self.stride + kernel_height, col * self.stride:col * 
                                                         self.stride + kernel_width, :] * dZ[:, row:row + 1, col:col + 1, 
                                                                                             np.newaxis, chan], axis=0)
        #we hold everything in gradients and then get the dX value and reutnr that 
        self.gradients["W"] = dW
        self.gradients["b"] = dB
        dX = paddingdX[:, self.pad[0]:in_rows + self.pad[0], self.pad[1]:in_cols + self.pad[1], :]
        return dX

class Pool2D(Layer):
    """Pooling layer, implements max and average pooling."""

    def __init__(
        self,
        kernel_shape: Tuple[int, int],
        mode: str = "max",
        stride: int = 1,
        pad: Union[int, Literal["same"], Literal["valid"]] = 0,
    ) -> None:

        if type(kernel_shape) == int:
            kernel_shape = (kernel_shape, kernel_shape)

        self.kernel_shape = kernel_shape
        self.stride = stride

        if pad == "same":
            self.pad = ((kernel_shape[0] - 1) // 2, (kernel_shape[1] - 1) // 2)
        elif pad == "valid":
            self.pad = (0, 0)
        elif isinstance(pad, int):
            self.pad = (pad, pad)
        else:
            raise ValueError("Invalid Pad mode found in self.pad.")

        self.mode = mode

        if mode == "max":
            self.pool_fn = np.max
            self.arg_pool_fn = np.argmax
        elif mode == "average":
            self.pool_fn = np.mean

        self.cache = {
            "out_rows": [],
            "out_cols": [],
            "X_pad": [],
            "p": [],
            "pool_shape": [],
        }
        self.parameters = {}
        self.gradients = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: use the pooling function to aggregate local information
        in the input. This layer typically reduces the spatial dimensionality of
        the input while keeping the number of feature maps the same.

        As with all other layers, please make sure to cache the appropriate
        information for the backward pass.

        Parameters
        ----------
        X  input array of shape (batch_size, in_rows, in_cols, channels)

        Returns
        -------
        pooled array of shape (batch_size, out_rows, out_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        # we want to use the kernel and X to get the row heights and widths
        #same as we did in Conv2D
        exnum, rin, cin, chin = X.shape
        k_height, k_width = self.kernel_shape
        chin = 1
        #this time around we are goiong to us the im2col and col2im method because that is faster
        channel_diff = []
        #here we set up our padding matrix with the 4 elements
        padding = (self.pad[0], self.pad[0], self.pad[1], self.pad[1])
        #similar to our conv2d we have the out rows and cols
        self.out_rows = out_rows = (rin + padding[0] + padding[1] - k_height) // self.stride + 1
        self.out_cols = out_cols = (cin + padding[2] + padding[3] - k_width) // self.stride + 1
        #the only difference is now we want to pool everything and we do so as follows:
        X_pool = np.zeros((X.shape[0], out_rows, out_cols, X.shape[3]))
        #now we iterate through all the channels and we want to get a slicve of the X according to the
        #channel that we are in and then we use im2col to rearrange the cols accordingly
        for chan in range(X.shape[3]):
            partofX = np.expand_dims(X[:, :, :, chan], axis=-1)
            colX, padding = im2col(partofX, self.kernel_shape, self.stride, self.pad)
            #we now call the poll function accordingly on the columns of X and then we 
            #will set up the gradient
            pool = self.pool_fn(colX, axis=0)
            grads = np.zeros(colX.shape)
            #based off the poll function we will adjust the gradients accordingly
            if self.pool_fn == np.max:
                grads[np.argmax(colX, axis=0), np.arange(colX.shape[1])] = 1
            elif self.pool_fn == np.mean:
                grads += 1/grads.shape[0]
            #here we add to the channel gradients and we resetup our pooling shape so that we 
            #can make this our final X_pool, which we will return after iterating through all channels
            channel_diff.append(grads)
            pool = pool.reshape(chin, out_rows, out_cols, exnum)
            pool = pool.transpose(3, 1, 2, 0)
            X_pool[:, :, :, chan] = pool.reshape(X.shape[0], out_rows, out_cols)

        # implement the forward pass
        # cache any values required for backprop
        self.cache["X"] = X
        self.cache["channel_diff"] = channel_diff
        ### END YOUR CODE ###

        return X_pool

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        """Backward pass for pooling layer.

        Parameters
        ----------
        dLdY  gradient of loss with respect to the output of this layer
              shape (batch_size, out_rows, out_cols, channels)

        Returns
        -------
        gradient of loss with respect to the input of this layer
        shape (batch_size, in_rows, in_cols, channels)
        """
        ### BEGIN YOUR CODE ###
        #first we want to pull out our value for X and channel grads and just as we did in all the other
        #function calls we want to use the X.shape to set up parameters
        X = self.cache["X"]
        channel_diff = self.cache["channel_diff"]
        nex, rin, cin, chin = X.shape
        #here now we want to use the X shape to make a matrix of 0s for the X and dX analysis
        X_t = np.zeros((X.shape[0], X.shape[1], X.shape[2], 1))
        dX = np.zeros(X.shape)
        #as before we iterate through all the channels and we want to get the gradient components
        for chan in range(chin):
            #use dLDY to get the outgoing gradient and reshape it and then we get the col vector 
            #once we get all this we use the col2im function to reshape everything according to the kernel
            #desired shape and then we reshape it and put it in dX, and once we have done this for all
            #the channels we return our dX
            diffout = np.expand_dims(dLdY[:, :, :, chan], axis=-1)
            diffout = diffout.transpose(3, 1, 2, 0).reshape(1, -1)
            diffcol = channel_diff[chan] * diffout
            diffp = col2im(diffcol, X_t, (self.kernel_shape[0], 
                                          self.kernel_shape[1], 1, 1), self.stride,
                                            (self.pad[0], self.pad[0], self.pad[1], 
                                             self.pad[1])).transpose(0, 2, 3, 1)
            dX[:, :, :, chan] = diffp.reshape(X.shape[0], X.shape[1], -1)
        return dX

class Flatten(Layer):
    """Flatten the input array."""

    def __init__(self, keep_dim: str = "first") -> None:
        super().__init__()

        self.keep_dim = keep_dim
        self._init_params()

    def _init_params(self):
        self.X = []
        self.gradients = {}
        self.parameters = {}
        self.cache = {"in_dims": []}

    def forward(self, X: np.ndarray, retain_derived: bool = True) -> np.ndarray:
        self.cache["in_dims"] = X.shape

        if self.keep_dim == -1:
            return X.flatten().reshape(1, -1)

        rs = (X.shape[0], -1) if self.keep_dim == "first" else (-1, X.shape[-1])
        return X.reshape(*rs)

    def backward(self, dLdY: np.ndarray) -> np.ndarray:
        in_dims = self.cache["in_dims"]
        dX = dLdY.reshape(in_dims)
        return dX
