def conv2d_output(size, kernel_size=3, stride=1):
    if size <= kernel_size:
        raise ValueError(
            'Creating conv layer with size {} and kernel size {} is impossible'
                .format(size, kernel_size))
    return (size - kernel_size) // stride + 1