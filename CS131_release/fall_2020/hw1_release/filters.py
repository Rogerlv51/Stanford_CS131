"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    # 首先需要对kernel进行flip，根据公式定义
    kernel = np.flip(kernel)

    # 遍历每一个像素点计算，原本应该做一个padding的操作，但是有更巧妙的方法
    for i in range(Hi):
        for j in range(Wi):
            sum = 0
            for m in range(Hk):
                for n in range(Wk):
                    # 如果当前要计算的像素点超出了kernel的范围就跳过即可，巧妙避开了padding的操作
                    if i+m-1<0 or j+n-1<0 or i+m-1>=Hi or j+n-1>=Wi:
                        continue
                    sum += image[i+m-1][j+n-1] * kernel[m][n]
            out[i][j] = sum
            
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE 这里为了方便直接使用numpy的padding函数，当然也可以先创建一个全0数组，然后填充image
    out = np.zeros((H+2*pad_height,W+2*pad_width))
    for i in range(H):
        for j in range(W):
            out[pad_height+i][pad_width+j] = image[i][j]
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    kernel = np.flip(kernel)
    image = zero_pad(image, Hk//2, Wk//2)
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = np.sum(np.multiply(image[i:i+Hk,j:j+Wk],kernel))
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    g = np.flip(g)    # 把模板翻转下，再调用之前写好的卷积函数即可
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    
    g = g - np.mean(g)
    g = np.flip(g)
    out = conv_fast(f, g)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    # 值得注意的是，这里的归一化是对每一个滑动窗口内的区域图像进行归一化，而不是对整个图像进行归一化
    g = (g - np.mean(g))/np.std(g)
    
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    height = Hg//2
    width = Wg//2
    image = zero_pad(f, height, width)
    for i in range(Hf):
        for j in range(Wf):
            sub_image = (image[i:i+Hg,j:j+Wg] - np.mean(image[i:i+Hg,j:j+Wg]))/np.std(image[i:i+Hg,j:j+Wg])
            out[i,j] = np.sum(np.multiply(sub_image,g))
    ### END YOUR CODE

    return out
