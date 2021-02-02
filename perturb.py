'''
Author: Ayrton San Joaquin
February 2021
'''

import torch
import torch.nn.functional as F

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import os
from os.path import join
import numpy as np
from tqdm import tqdm

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

def tv_norm(input, tv_beta):
    '''
    Computes the Total Variation (TV) denoising term
    Parameters: input image, tv_beta
    Returns: TV term
    '''
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def save(mask, img, blurred, out, plot=True):
    '''
    Creates, saves, and optionally, plots the images
    Parameters: Mask, original image, blurred image, output directory
    plot - plot the image
    Returns: void
    '''
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    
    heatmap = np.float32(heatmap) / 255
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbed = np.multiply(1 - mask, img) + np.multiply(mask, blurred)	

    perturbed = Image.fromarray(np.uint8(255 * perturbed))
    perturbed.save(join(out,'perturbed.png'))

    heatmap = Image.fromarray(np.uint8(255 * heatmap))
    heatmap.save(join(out,'heatmap.png'))
    
    # squeeze because grayscale image (1 color channel)
    mask = Image.fromarray(np.squeeze(np.uint8(255 * mask), axis=2))
    mask.save(join(out,'mask.png'))

    cam = Image.fromarray(np.uint8(255 * cam))
    cam.save(join(out,'cam.png'))

    # Plot images
    if plot:
      plt.figure()

      plt.subplot(131)
      plt.title('Original')
      plt.imshow(img * 255)
      plt.axis('off')

      plt.subplot(132)
      plt.title('Mask')
      plt.imshow(np.uint8(255 * mask))
      plt.axis('off')

      plt.subplot(133)
      plt.title('Perturbed Image')
      plt.imshow(np.uint8(255 * perturbed))
      plt.axis('off')

      plt.tight_layout()
      plt.show()


def upsample(image):
    return F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False).to(device)

def perturb(image, model, transforms, out_dir='/content/perturb_outputs', \
    tv_beta=3, lr=0.1, max_iter=100, l1_coeff=0.01, tv_coeff=0.02, is_numpy=False):
    '''
    Computes the mask via Stochastic Gradient Descent (SGD) and 
    applies perturbation onto Image as described by 
    Meaningful Perturbations (2018)

    Parameters:
    image - image to perturb
    model - Black-box model to be used
    transforms - affine transformations to preprocess the image
    out_dir - output directory where the resulting images will be saved
    tv_beta - degree of the Total Variation denoising norm
    lr - learning rate
    max_iter - the number of iterations for SGD
    l1_coeff - L1 regularization coefficient
    tv_coeff - TV coefficient (Lambda_2 in the paper)
    is_numpy - whether the image passed is already a numpy array

    Returns: void (calls the save function to create and save the resulting images)
    '''
  
    if is_numpy:
      original_img = image
    else:
      original_img = np.array(Image.open(image).convert('RGB').resize((224, 224)))

    blurred_img = cv2.GaussianBlur(original_img, (11, 11), 5)
    # generate mask
    # Sample from U[0,1)
    mask = torch.ones((1, 1, 28, 28), dtype = torch.float32, requires_grad=True, device=device)

    # image tensor
    img_tensor = transforms(original_img).unsqueeze(0).to(device)
    blurred_tensor = transforms(blurred_img).to(device)

    optimizer = torch.optim.Adam([mask], lr=lr)

    prob = torch.nn.Softmax(dim=1)(model(img_tensor))
    class_idx = np.argmax(prob.cpu().data.numpy())
    print( "Predicted class index: {}. Probability before perturbation: {}".format(class_idx, prob[0, class_idx]))

    for i in range(max_iter):
        upsampled_mask = upsample(mask)
        
        # Use the mask to perturbe the image
        perturbed_input = img_tensor.mul(upsampled_mask) + \
                            blurred_tensor.mul(1-upsampled_mask)
        
        # add some noise to the perturbed image for the model to learn from multiple masks
        noise = (torch.randn((1, 3, 224, 224), device=device))
        perturbed_input = perturbed_input + noise
        
        masked_idx = torch.nn.Softmax(dim=1)(model(perturbed_input))
        masked_prob = masked_idx[0, class_idx]

        loss = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
                tv_coeff * tv_norm(mask, tv_beta) + masked_prob

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mask.data.clamp_(0, 1)
        if i% 20 == 0:
            print('Loss: {}, Probability for target class {}'.format(loss, masked_prob))
    # Create directory
    if not os.path.exists(out_dir):
      os.mkdir(out_dir)

    save(upsample(mask), original_img, blurred_img, out_dir)