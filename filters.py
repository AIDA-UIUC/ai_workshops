"""Contains different filters for 2D convolutions
"""
import numpy as np


class Kernel:
    """General kernel parent class
    """

    def __init__(self):
        self.kernel = None

    def __call__(self):
        return self.kernel

    def __str__(self):
        return '\n'.join(str(row) for row in self.kernel)


class UnitImpulse(Kernel):
    """Defines the unit impulse or dirac delta kernel
    """

    def __init__(self, size=3):
        super().__init__()

        self.kernel = np.zeros((size, size))

        # impulse in the middle of the filter
        center = size // 2
        self.kernel[center, center] = 1


class BoxBlur(Kernel):
    """Defines the local average kernel
    """

    def __init__(self, size=3):
        super().__init__()

        self.kernel = np.ones((size, size))
        self.kernel /= self.kernel.sum()


class GaussianBlur(Kernel):
    """Defines the Gaussian blur kernel
    """

    def __init__(self, size, mu=0, std=1):
        super().__init__()
        self.mu = mu
        self.std = std
        self.size = size

        # create bivariate gaussian using separability
        x = self.normal_pdf()
        y = self.normal_pdf()
        self.kernel = np.outer(x, y)

    def normal_pdf(self):
        x = np.linspace(-self.std, self.std, self.size)
        return np.exp(-0.5 * ((x - self.mu) / self.std)**2) / np.sqrt(2 * np.pi * self.std**2)


class Prewitt(Kernel):
    """Defines the Prewitt gradient kernel (fixed size 3x3)
    """

    def __init__(self, mode):
        super().__init__()

        # again, using separability
        ones = np.ones(3)
        grad = np.array([1, 0 , -1])
        k = np.outer(ones, grad)

        # different edge detecting modes
        if mode not in ["horiz", "vert", "ldiag", "rdiag"]:
            raise NotImplementedError("Filter mode must be one of ['horiz', 'vert', 'ldiag', 'rdiag']")

        self.kernel = {
            "horiz": k,
            "vert": k.T,
            "ldiag": k + k.T,
            "rdiag": k - k.T
        }[mode]


class Sobel(Kernel):
    """Defines the Sobel gradient kernel (fixed size 3x3)
    """

    def __init__(self, mode):
        super().__init__()

        # once again, using separability
        grad = np.array([1, 0 , -1])
        avg = np.array([1, 2, 1])
        k = np.outer(avg, grad)

        # different edge detecting modes
        if mode not in ["horiz", "vert", "ldiag", "rdiag"]:
            raise NotImplementedError("Filter mode must be one of ['horiz', 'vert', 'ldiag', 'rdiag']")

        self.kernel = {
            "horiz": k,
            "vert": k.T,
            "ldiag": k + k.T,
            "rdiag": k - k.T
        }[mode]


class HighPass(Kernel):
    """Defines a basic 3x3 high pass filter
    """

    def __init__(self):
        super().__init__()

        self.kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ])


class Laplacian(Kernel):
    """Defines a basic 3x3 laplacian kernel (also a high pass filter)
    """

    def __init__(self):
        super().__init__()

        self.kernel = np.array([
            [0, -1,  0],
            [-1, 4, -1],
            [0, -1,  0]
        ])

