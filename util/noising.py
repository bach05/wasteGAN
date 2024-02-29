import torch
import logging
from torch import optim
from tqdm import tqdm

############################################################
###   NOISE MODELLING
############################################################

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", sched="exp"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        if sched=="exp":
            self.beta = self.exp_noise_schedule(noise_steps).to(device)
        if sched=="lin":
            self.beta = self.prepare_noise_schedule(noise_steps).to(device)
        
        print("beta: ",self.beta.shape)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        print("alpha_hat: ", self.alpha_hat.shape)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def exp_noise_schedule(self, steps = 999):
        val = torch.arange(0,self.noise_steps)
        rate = 5/steps
        return self.beta_start + self.beta_end*(1-torch.exp(-rate*val))

    def noise(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])
        eps = torch.randn_like(x)
        return sqrt_alpha_hat, sqrt_one_minus_alpha_hat, eps
        #return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).repeat(x.shape[1],x.shape[2], x.shape[3],1).permute(3,0,1,2)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).repeat(x.shape[1],x.shape[2], x.shape[3],1).permute(3,0,1,2)
        eps = torch.randn_like(x)
        #print("sqrt_one_minus_alpha_hat: ", sqrt_one_minus_alpha_hat.shape)
        #print("self.alpha_hat[t]", self.alpha_hat[t].shape)
        #print("torch.sqrt(1 - self.alpha_hat[t])", torch.sqrt(1 - self.alpha_hat[t]).shape)
        #print("t: ", t)
        #print("sqrt_alpha_hat: ", sqrt_alpha_hat.shape)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def denoise_sample(self, x, predicted_noise, mask, t=999):
        
        alpha = self.alpha[t]
        alpha_hat = self.alpha_hat[t]
        beta = self.beta[t]
        noise = torch.randn_like(x)
        
        x = 1 / torch.sqrt(alpha) * (x[mask] - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def train_sample(self, x, model, steps, args, labels=None):
        
        for i in reversed(range(steps)):
                t = (torch.ones(args.batch_size) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                alpha = self.alpha[t].repeat(x.shape[1],x.shape[2], x.shape[3],1).permute(3,0,1,2)
                alpha_hat = self.alpha_hat[t].repeat(x.shape[1],x.shape[2], x.shape[3],1).permute(3,0,1,2)
                beta = self.beta[t].repeat(x.shape[1],x.shape[2], x.shape[3],1).permute(3,0,1,2)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        return x

    def sample(self, model, n, labels, cfg_scale=3):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, self.img_size[0], self.img_size[1], self.img_size[2])).to(self.device)
            input = x.clone()
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t].repeat(x.shape[1],x.shape[2], x.shape[3],1).permute(3,0,1,2)
                alpha_hat = self.alpha_hat[t].repeat(x.shape[1],x.shape[2], x.shape[3],1).permute(3,0,1,2)
                beta = self.beta[t].repeat(x.shape[1],x.shape[2], x.shape[3],1).permute(3,0,1,2)
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x) #for t=0, zero noise
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x, input


def applyNoise(diff, patch, mask, device, time=0, std_base=0.8, scheduler_time_constant = 0.007, offset = 1.5):
    
    '''
    np.random.seed()

    mask = mask < 0
    mask = mask.repeat([1,3,1,1])
    std_scale_factor = (offset - math.exp(-scheduler_time_constant*time)) / offset
    std = std_base * std_scale_factor

    print("std: ", std_scale_factor, std)

    gaussian = (torch.randn(size=patch.shape)*std).to(device)
    patch[mask] = patch[mask] + gaussian[mask]
    '''

    mask = mask < 0
    mask = mask.repeat([1,3,1,1])
    sqrt_alpha_hat, sqrt_one_minus_alpha_hat, eps = diff.noise(patch, time)

    patch[mask] = sqrt_alpha_hat * patch[mask] + sqrt_one_minus_alpha_hat * eps[mask]
    noise = eps[mask]
    return patch