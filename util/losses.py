import torch
import torch.nn as nn
from models.resnet import basicSemSegModel
import copy
from torchvision.transforms.functional import rgb_to_grayscale
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import matplotlib.gridspec as gridspec
import numpy as np
from util.diffusion_utils import normalizeRGB

def rescaledMSE(output, target, alpha=5):
    loss = torch.mean((output - target)**2)
    loss = 1 / (-alpha *(loss-1)) - 1 / alpha
    return loss

class resnetPercLoss(object):
    def __init__(self, weights, device):
        self.model = basicSemSegModel(device=device)
        self.model.load_state_dict(torch.load(weights))
        self.model.eval().to(device)
        self.mse = nn.MSELoss()

    def __call__(self, input, target):
        _, feat_input = self.model(input)
        _, feat_target = self.model(target)
        return self.mse(feat_input, feat_target)


class MutualInformation(nn.Module):

	def __init__(self, sigma=0.1, in_shape=[1,256,1], normalize=True, device=""):
		super(MutualInformation, self).__init__()

		#in_shape = [batch, W*H, channels]

		self.sigma = sigma
		self.num_bins = in_shape[1]
		self.normalize = normalize
		self.epsilon = 1e-10
		self.device = device

		self.bins = (torch.linspace(0, 255, self.num_bins, device=device, requires_grad=False)).float()
		self.bins = self.bins.unsqueeze(0).unsqueeze(2).repeat(in_shape[0],1,in_shape[2])
		print("bins: ", self.bins.shape)

	def marginalPdf(self, values):

		residuals = values - self.bins
		kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
		
		pdf = torch.mean(kernel_values, dim=1)
		normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
		pdf = pdf / normalization
		
		return pdf, kernel_values


	def jointPdf(self, kernel_values1, kernel_values2):

		joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
		normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
		pdf = joint_kernel_values / normalization

		return pdf


	def getMutualInformation(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W
			return: scalar
		'''

		# Torch tensors for images between (0, 1)
		input1 = input1*255
		input2 = input2*255

		B, C, H, W = input1.shape
		assert((input1.shape == input2.shape))

		x1 = input1.view(B, H*W, C)
		x2 = input2.view(B, H*W, C)
		
		pdf_x1, kernel_values1 = self.marginalPdf(x1)
		pdf_x2, kernel_values2 = self.marginalPdf(x2)
		pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

		H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
		H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
		H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

		mutual_information = H_x1 + H_x2 - H_x1x2
		
		if self.normalize:
			mutual_information = 2*mutual_information/(H_x1+H_x2)

		return mutual_information


	def forward(self, input1, input2):
		'''
			input1: B, C, H, W
			input2: B, C, H, W
			return: scalar
		'''
		return self.getMutualInformation(input1, input2)


class MutualInformation2(nn.Module):

	def __init__(self, num_classes, num_bins=256, device="cuda", fig_name="pJoint", cat=None, use_ema=True, ema_w=0.99):
		super(MutualInformation2, self).__init__()

		self.num_classes = num_classes
		self.num_bins = num_bins
		self.device = device
		self.fig_name = fig_name
		self.categories = cat
		self.use_ema = use_ema
		self.ema_w = ema_w
		#self.categories.insert(0, "0")

		self.ema_pjoint = torch.zeros((self.num_classes, self.num_bins)).to(self.device).long()
		self.first_update = True

	def getProbs(self, image, mask):

		#image is [B,C,H,W]
		B, C, H, W = image.shape
		# compare = torch.Tensor(range(self.num_classes)).to(self.device)
		# compare = compare.unsqueeze(1).unsqueeze(2).repeat(1, H, W)
		# compare = compare.unsqueeze(0).repeat(B, 1, 1, 1)
		# mask = torch.sum(torch.mul(mask,compare),dim=1).unsqueeze(1)
		#print("mask: ", mask.shape)
		mask = torch.argmax(mask.detach(), dim=1).unsqueeze(1)

		COM = torch.zeros((self.num_classes, self.num_bins)).to(self.device).long()

		image_flat = copy.deepcopy(image.detach())
		image_flat = rgb_to_grayscale(image_flat)
		grid_img = make_grid(image_flat, nrow=4)
		#print("image flat: ", image_flat.shape)
		mask_flat = copy.deepcopy(mask.detach())
		grid_mask = make_grid(mask_flat/self.num_classes, nrow=4)

		#convert to uint8 if necessary 
		if image_flat.dtype != torch.uint8:
			#image_rgb is double
			image_flat = (image_flat * 255).to(torch.uint8)
			grid_img = make_grid(image_flat, nrow=4)
			#print("uint8 image rgb: ", torch.unique(image_flat))
		if mask_flat.dtype != torch.uint8:
			#image_rgb is double
			mask_flat = (mask_flat).to(torch.uint8)
			#print("uint8 image rgb: ", torch.unique(image_flat))
	
		#flatten the image and the mask
		image_flat = torch.flatten(image_flat)
		mask_flat = torch.flatten(mask_flat)
		#print("mask flat:", torch.unique(mask_flat))

		#for each class label, compute the cumulative probability function
		for i in range(self.num_classes):
			filter_mask = (mask_flat == i) # 1 where mask==i, 0 otherwise
			#print("filter mask: ", filter_mask)
			real_zeros = image_flat.shape[0] - torch.count_nonzero(image_flat) #real number of pixels with value 0
			filter_image = filter_mask * image_flat #retain only pixels values that corresponds to label i
			out, counts = torch.unique(filter_image, return_counts=True) #count how many pixels of each value are still present 
			#print("out:", out.shape)
			#print("filter image:", filter_image.shape)
			#print(COM.shape)
			#subCOM = torch.index_select(COM[i,:], dim=0, index=out.int())
			#subCOM = counts
			COM[i,out.long()] = counts #report the counts on the cumulative distirbution
			COM[i,0] = real_zeros

		#p_joint =  COM.float() / (H*W*B)

		if self.use_ema:
			if self.first_update:
				self.ema_pjoint = COM.float() / (H*W*B)
				self.first_update = False
			else:
				self.ema_pjoint = self.ema_pjoint*self.ema_w + ( COM.float() / (H*W*B) )*(1-self.ema_w)

			p_joint = self.ema_pjoint
		else:
			p_joint =  COM.float() / (H*W*B)
		
		p_img = torch.sum(p_joint, dim=0) 
		p_mask = torch.sum(p_joint, dim=1)
		
		#print("p_mask: ", torch.min(p_mask, ))
		eps = 10**-9
		p_mask_img = p_joint / (p_img.unsqueeze(0).repeat(5,1)+eps)
		#print("pmask shape: ", p_mask.unsqueeze(1).repeat(1,256).shape)
		#print("pimg shape: ", p_img.unsqueeze(0).repeat(5,1).shape)
		#print("pjoint shape: ", p_joint.shape)
		p_img_mask = p_joint / (p_mask.unsqueeze(1).repeat(1,256)+eps)

		gs = gridspec.GridSpec(2, 2)

		fig=plt.figure()
		ax = plt.subplot(gs[0, 0]) # row 0, col 0
		plt.imshow(grid_img.detach().permute(1,2,0).cpu().numpy() )

		ax = plt.subplot(gs[0, 1]) # row 0, col 1
		plt.imshow(grid_mask.detach().permute(1,2,0).cpu().numpy() )

		if self.fig_name == "pMask|Img":
			distribution = p_mask_img
		if self.fig_name == "pJoint":
			distribution = p_joint

		ax = plt.subplot(gs[1, :]) # row 1, span all columns
		plt.imshow(distribution.detach().cpu().numpy(), aspect='auto',  interpolation='nearest' )
		if self.categories is not None:
			ax.set_yticklabels(self.categories)
		# plt.show()

		return p_joint, p_img, p_mask, p_img_mask, p_mask_img, fig


	def getProbsRGB(self, image, mask):

		#image is [B,C,H,W]
		B, C, H, W = image.shape
		# compare = torch.Tensor(range(self.num_classes)).to(self.device)
		# compare = compare.unsqueeze(1).unsqueeze(2).repeat(1, H, W)
		# compare = compare.unsqueeze(0).repeat(B, 1, 1, 1)
		# mask = torch.sum(torch.mul(mask,compare),dim=1).unsqueeze(1)
		#print("mask: ", mask.shape)
		mask = torch.argmax(mask.detach(), dim=1).unsqueeze(1)

		COM = torch.zeros((C, self.num_classes, self.num_bins)).to(self.device).long()

		image_flat = copy.deepcopy(image.detach())
		#image_flat = rgb_to_grayscale(image_flat)
		grid_img = make_grid(image_flat, nrow=4)
		#print("image flat: ", image_flat.shape)
		mask_flat = copy.deepcopy(mask.detach())
		grid_mask = make_grid(mask_flat/self.num_classes, nrow=4)

		#convert to uint8 if necessary 
		if image_flat.dtype != torch.uint8:
			#image_rgb is double
			image_flat = (image_flat * 255).to(torch.uint8)
			grid_img = make_grid(image_flat, nrow=4)
			#print("uint8 image rgb: ", torch.unique(image_flat))
		if mask_flat.dtype != torch.uint8:
			#image_rgb is double
			mask_flat = (mask_flat).to(torch.uint8)
			#print("uint8 image rgb: ", torch.unique(image_flat))
	
		#flatten the image and the mask
		image_flat = torch.flatten(image_flat.permute(1,0,2,3), start_dim=1)
		mask_flat = torch.flatten(mask_flat)
		#print("image flat: ", image_flat.shape) # 3x65536
		#print("mask flat:", torch.unique(mask_flat))
		#print("image flat:", torch.unique(image_flat))

		#for each class label, compute the cumulative probability function
		for c in range(C):
			for i in range(self.num_classes):
				filter_mask = (mask_flat == i) # 1 where mask==i, 0 otherwise
				#print("filter mask: ", filter_mask)
				real_zeros = filter_mask * (image_flat[c] == 0)
				real_zeros = torch.sum(real_zeros)
				#print(f"real zeros ch {c} class {i}: {real_zeros} / { image_flat.shape[1]}") #3
				filter_image = filter_mask * image_flat[c] #retain only pixels values that corresponds to label i
				#print("filter image:", filter_image.shape)
				out, counts = torch.unique(filter_image, return_counts=True) #count how many pixels of each value are still present 
				# print("out:", out.shape) #3 x distinct values count
				# print("out:", out) #3 x distinct values count
				# print("counts:", counts.shape) #3 x distinct values count
				#print(f"counts ch {c} class {i}: {counts}") #3 x distinct values count
				#print("filter image:", filter_image.shape)
				#print(COM.shape)
				#subCOM = torch.index_select(COM[i,:], dim=0, index=out.int())
				#subCOM = counts
				COM[c, i, out.long()] = counts #report the counts on the cumulative distirbution
				COM[c, i, 0] = real_zeros
				#print(COM[c, i, :])
				#print("**********************")
			
		#print("COM: ", COM.shape)

		if self.use_ema:
			if self.first_update:
				self.ema_pjoint = COM.float() / (H*W*B)
				self.first_update = False
			else:
				self.ema_pjoint = self.ema_pjoint*self.ema_w + ( COM.float() / (H*W*B) )*(1-self.ema_w)

			p_joint = self.ema_pjoint
		else:
			p_joint =  COM.float() / (H*W*B)

		#print("P(mask , img): ", torch.sum(torch.sum(p_joint, dim=1),dim=1))
		p_img = torch.sum(p_joint, dim=1) 
		p_mask = torch.sum(p_joint, dim=2)
		eps = 10**-9
		#print("p_img: ", p_img.shape) 
		#print("p_mask: ", p_mask.shape) 
		#print("p_mask: ", p_mask) 
		#print("p_joint: ", p_joint.shape)
		p_mask_img = p_joint / (p_img.unsqueeze(1).repeat(1,self.num_classes,1)+eps)
		#print("pmask shape: ", p_mask.unsqueeze(1).repeat(1,256).shape)
		#print("pimg shape: ", p_img.unsqueeze(0).repeat(5,1).shape)
		#print("pjoint shape: ", p_joint.shape)
		p_img_mask = p_joint / (p_mask.unsqueeze(2).repeat(1,1,256)+eps)

		# print("P(mask | img): ", p_mask_img.shape)
		# print("P(mask | img) sum: ", torch.sum(p_mask_img, dim=1).shape)
		# print("P(mask | img) sum: ", torch.sum(p_mask_img, dim=1))

		#print("P(img | mask): ", p_mask_img.shape)
		#print("P(img | mask) sum: ", torch.sum(p_img_mask, dim=2))
		#print("P(img | mask) sum shape: ", torch.sum(p_img_mask, dim=2).shape)
		#print("P(img | mask) uni: ", torch.unique(p_img_mask, dim=2))

		gs = gridspec.GridSpec(2, 2)

		fig=plt.figure()
		ax = plt.subplot(gs[0, 0]) # row 0, col 0
		plt.imshow(grid_img.detach().permute(1,2,0).cpu().numpy() )
		ax = plt.subplot(gs[0, 1]) # row 0, col 1
		plt.imshow(grid_mask.detach().permute(1,2,0).cpu().numpy() )

		# ax = plt.subplot(gs[0, :]) # row 0, col 0
		# distribution = normalizeRGB((p_joint.unsqueeze(0))).squeeze()
		# distribution = make_grid(distribution.unsqueeze(1), nrow=3, pad_value=1.0)
		# plt.imshow(distribution.detach().permute(1,2,0).cpu().numpy(),  aspect='auto',  interpolation='nearest' )


		if self.fig_name == "pMask|Img":
			distribution = p_mask_img
		if self.fig_name == "pImg|Mask":
			distribution = p_img_mask
		if self.fig_name == "pJoint":
			distribution = p_joint

		distribution = normalizeRGB(distribution.unsqueeze(0)).squeeze()
		#print("ditribution sum: ", torch.sum(distribution))

		ax = plt.subplot(gs[1, :]) # row 1, span all columns
		distribution = make_grid(distribution.unsqueeze(1), nrow=3, pad_value=1.0)

		distribution = distribution.permute(1,2,0)
		plt.imshow(distribution.detach().cpu().numpy(), aspect='auto',  interpolation='nearest' )
		if self.categories is not None:
			ax.set_yticklabels(self.categories)
		#plt.show()

		return p_joint, p_img, p_mask, p_img_mask, p_mask_img, fig



	def forward(self, input, mask, rgb=False):
		'''
			input1: B, C, H, W
			input2: B, C, H, W
			return: scalar
		'''
		input = normalizeRGB(input, use_int8=True)
		#print("input ", torch.unique(input))
		if rgb:
			p_joint, p_img, p_mask, p_img_mask, p_mask_img, fig = self.getProbsRGB(input, mask)
		else:
			p_joint, p_img, p_mask, p_img_mask, p_mask_img, fig = self.getProbs(input, mask)
		
		return {"pJoint":p_joint, "pImg":p_img, "pMask":p_mask, "pImg|Mask":p_img_mask, "pMask|Img":p_mask_img, "fig":fig}


class Entropy(nn.Module):
	def __init__(self, eps=10**-9, device="cuda"):
		super(Entropy, self).__init__()

		self.device = device
		self.eps = eps

	def forward(self, jointP):

		logP = torch.log2(jointP+self.eps)
		entropy = -torch.sum(torch.mul(jointP,logP))

		return entropy
		
class Bhattacharyya(nn.Module):
	def __init__(self, device="cuda"):
		super(Bhattacharyya, self).__init__()

		self.device = device

	def forward(self, p1, p2, rgb=False):

		if rgb:
			sum = torch.sum(torch.sum(torch.sqrt(torch.mul(p1,p2)), dim=1), dim=1)
			distance = torch.mean(sum) #average the 3 channels
		else:
			distance = torch.sum(torch.sqrt(torch.mul(p1,p2)))
		return distance

class BhattacharyyaMgI(nn.Module):
	def __init__(self, device="cuda", eps=0.001):
		super(BhattacharyyaMgI, self).__init__()

		self.device = device
		self.eps = eps

	def forward(self, p1, p2, rgb=False):

		if rgb:
			#print("p1: ", p1.shape)
			sum = torch.sum(torch.sqrt(torch.mul(p1,p2)), dim=2)
			#print("sum: ", sum.shape) 
			distance = torch.mean(sum, dim=1)
			#print("distance: ", distance)
		else:
			#print("p1: ", p1.shape)
			sum = torch.sum(torch.sqrt(torch.mul(p1,p2)), dim=1)
			#sum = sum[1:-1] 
			#print("sum:", sum)
			#print("Distance bat: ", torch.mean(distance))

		return torch.mean(distance)

class BhattacharyyaIgM(nn.Module):
	def __init__(self, device="cuda", eps=0.001):
		super(BhattacharyyaIgM, self).__init__()

		self.device = device
		self.eps = eps

	def forward(self, p1, p2, rgb=False):

		if rgb:
			#print("p1: ", p1.shape)
			sum = torch.sum(torch.sqrt(torch.mul(p1,p2)), dim=1)
			#print("sum: ", sum.shape) 
			distance = torch.mean(sum, dim=1)
			#print("distance: ", distance)
		else:
			#print("p1: ", p1.shape)
			sum = torch.sum(torch.sqrt(torch.mul(p1,p2)), dim=1)
			#sum = sum[1:-1] 
			#print("sum:", sum)
			#print("Distance bat: ", torch.mean(distance))

		return torch.exp(2*torch.mean(distance))-1
		#return torch.mean(distance)

class SharpLoss(nn.Module):
	def __init__(self, device="cuda"):
		super(SharpLoss, self).__init__()
		self.device = device
		self.Gx = torch.Tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).unsqueeze(0).unsqueeze(1).to(device)
		self.Gy = torch.Tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).unsqueeze(0).unsqueeze(1).to(device)

	def forward(self, fake_img, real_img):

		img = rgb_to_grayscale(fake_img)
		gx = nn.functional.conv2d(img, self.Gx)
		gy = nn.functional.conv2d(img, self.Gy)

		gnorm = torch.sqrt(gx**2 + gy**2)
		sharpness_fake = torch.mean(gnorm)

		img = rgb_to_grayscale(real_img)
		gx = nn.functional.conv2d(img, self.Gx)
		gy = nn.functional.conv2d(img, self.Gy)

		gnorm = torch.sqrt(gx**2 + gy**2)
		sharpness_real = torch.mean(gnorm)

		return abs(sharpness_real - sharpness_fake)


class ClassBalance(nn.Module):
	def __init__(self, num_classes, device="cuda", ema_w = 0.99):
		super(ClassBalance, self).__init__()
		self.device = device
		self.num_classes = num_classes
		self.uniform = torch.zeros(num_classes).to(device) + 1/num_classes
		self.norm_factor = 1/self.num_classes
		self.ema_w = ema_w

		self.class_distribution = None

	def updateClassDistribution(self, generated_masks):

		hist = torch.zeros(self.num_classes, dtype=torch.int64).to(self.device)
		mask_classes = torch.argmax(generated_masks, dim=1)
		unique_values, counts = torch.unique(mask_classes, return_counts=True)
		hist[unique_values] = counts
		if self.class_distribution is None:
			self.class_distribution = hist / (mask_classes.shape[0] * mask_classes.shape[1] * mask_classes.shape[2])
		else:
			self.class_distribution = self.class_distribution * self.ema_w + (1-self.ema_w) * hist / (
						mask_classes.shape[0]*mask_classes.shape[1]*mask_classes.shape[2])

	def forward(self, generated_masks):
		self.updateClassDistribution(generated_masks)
		return torch.norm((self.class_distribution - self.norm_factor) / (1-self.norm_factor)), self.class_distribution