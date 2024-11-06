import torch
from q2_sampler import svhn_sampler
from q2_model import Critic, Generator
from torch import optim
from torchvision.utils import save_image



def lp_reg(x, y, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    batchsize, C, H, W = x.shape
    alpha = torch.rand((batchsize, 1, 1, 1)).repeat(1, C, H, W).to(device)
    z = alpha*x + (1-alpha)*y
    critic_z = critic(z)

    gradient = torch.autograd.grad(
        inputs=z,
        outputs=critic_z,
        grad_outputs=torch.ones_like(critic_z),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = torch.norm(gradient, p=2, dim=1)
    gradient_penalty = torch.clamp(gradient_norm -1, min=0, max=float('inf'))**2

    return torch.mean(gradient_penalty)


def vf_wasserstein_distance(p, q, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    cp = critic(p)
    cq = critic(q)
    w = torch.mean(cp) - torch.mean(cq)
    return w



if __name__ == '__main__':
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_iter = 15000 # N training iterations
    n_critic_updates = 5 # N critic updates per generator update
    lp_coeff = 10 # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100
    losses = {'G': [], 'D': []}

    train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)

    generator = Generator(z_dim=z_dim).to(device)
    critic = Critic().to(device)

    optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
    optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    # COMPLETE TRAINING PROCEDURE
    train_iter = iter(train_loader)
    valid_iter = iter(valid_loader)
    test_iter  = iter(test_loader)
    for i in range(n_iter):
        generator.train()
        critic.train()
        for _ in range(n_critic_updates):
            try:
                data = next(train_iter)[0].to(device)
            except Exception:
                train_iter = iter(train_loader)
                data = next(train_iter)[0].to(device)
            true_X = data
            z = torch.randn(64, z_dim, device=device)
            fake_X = generator(z)

            c_loss = - vf_wasserstein_distance(true_X, fake_X, critic) + lp_coeff*lp_reg(true_X, fake_X, critic)

            optim_critic.zero_grad()
            c_loss.backward()
            optim_critic.step()
            losses['D'].append(c_loss)
            

        #####
        # train the generator model here
        #####
        z = torch.randn(64, z_dim, device=device)
        fake_X = generator(z)

        g_loss = -torch.mean(critic(fake_X))
        
        optim_generator.zero_grad()
        g_loss.backward()
        optim_generator.step()
        
        # Record loss
        losses['G'].append(g_loss)
        #####
        print(f"===== Loss G is : {g_loss:.3f} =====>")
        
        #save model
        torch.save(generator, 'G.pkl')
        torch.save(critic, 'D.pkl')


        # Save sample images 
        if i % 100 == 0:
            z = torch.randn(64, z_dim, device=device)
            imgs = generator(z)
            save_image(imgs, f'imgs_{i}.png', normalize=True, value_range=(-1, 1))


    # COMPLETE QUALITATIVE EVALUATION
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load("G.pkl").cuda()
    model.to(device)
    # switch to evaluate mode
    model.eval()

    try:
        os.makedirs("/content/gdrive/MyDrive/A3")
    except OSError:
        pass

    with torch.no_grad():
        for i in range(64):
            noise = torch.randn(64, 100, device=device)
            fake = model(noise)
            vutils.save_image(fake.detach(), f"/content/gdrive/MyDrive/A3/fake_{i:04d}.png", normalize=True)
        print("The fake image has been generated!")