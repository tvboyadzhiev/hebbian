import torch


class PCAColorJitter(object):
    imagenet_pca = {
        # The eigenvalues are organised as column vector
        'eigval': torch.Tensor([[0.2175], [0.0188], [0.0045]]),

        # The eigenvectors are assumed to be column vectors
        'eigvec': torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])
    }

    def __init__(self, alpha_std, pca_params):
        self.alpha_std = alpha_std
        self.pca_params = pca_params

    def __call__(self, img):
        if self.alpha_std == 0:
            return img

        return img + torch.matmul(
                self.pca_params['eigvec'],
                torch.randn(3, 1) * self.alpha_std * self.pca_params['eigval']
            ).view(3, 1, 1)
