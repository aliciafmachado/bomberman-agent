def conv2d_output(size, kernel_size=3, stride=1):
    return (size - kernel_size) // stride + 1