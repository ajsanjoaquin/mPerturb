import torch
import torch.nn.functional as F

from PIL import Image
import cv2
import matplotlib.pyplot as plt

import os
from os.path import join, basename, splitext
import numpy as np
from tqdm import tqdm

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

def tv_norm(input, tv_beta):
    '''
    Computes the Total Variation (TV) denoising term
    '''
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def postprocess(mask):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
    return 1 - mask

def save(mask, img, blurred, out, filename, plot=True):
    '''
    Creates, saves, and optionally, plots the images
    '''
    mask = postprocess(mask)
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbed = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    perturbed_img = Image.fromarray(np.uint8(255 * perturbed))
    perturbed_img.save(join(out, filename + 'perturbed.png'))

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img.save(join(out, filename + 'heatmap.png'))
    
    # squeeze because grayscale image (1 color channel)
    mask = np.squeeze(np.uint8(255 * mask), axis=2)
    mask_img = Image.fromarray(mask)
    mask_img.save(join(out, filename + 'mask.png'))

    cam = Image.fromarray(np.uint8(255 * cam))
    cam.save(join(out, filename + 'cam.png'))

    # Plot images
    if plot:
        plt.figure()

        plt.subplot(131)
        plt.title('Original')
        plt.imshow(np.uint8(img * 255))
        plt.axis('off')

        plt.subplot(132)
        plt.title('Mask')
        plt.imshow(mask, cmap='gray')
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
    tv_beta=3, lr=0.2, max_iter=100, l1_coeff=0.01, tv_coeff=0.02, \
    plot=True):
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
    plot - plot images

    Returns: void (calls the save function to create and save the resulting images)
    '''
    original_img = np.array(Image.open(image).convert('RGB').resize((224, 224)))
    filename = splitext(basename(image))[0]

    blurred_img = cv2.GaussianBlur(np.float32(original_img / 255), (11, 11), 5)
    # generate mask
    mask = torch.ones((1, 1, 28, 28), dtype = torch.float32, requires_grad=True, device=device)

    img_tensor = transforms(original_img).unsqueeze(0).to(device)
    blurred_tensor = transforms(blurred_img).to(device)

    optimizer = torch.optim.Adam([mask], lr=lr)

    prob = torch.nn.Softmax(dim=1)(model(img_tensor))
    class_idx = np.argmax(prob.cpu().data.numpy())
    print(f'Predicted class index: {class_idx}. Probability before perturbation: { prob[0, class_idx]}')

    for i in range(max_iter):
        upsampled_mask = upsample(mask)
        
        # perturb the image with mask
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
            print(f'Iteration {i}/{max_iter}, Loss: {loss}, Probability for target class {masked_prob}, Predicted label{class_idx}')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    save(upsample(mask), original_img, blurred_img, out_dir, filename, plot)

    # Mask can be used further, so return
    mask = postprocess(mask)
    return mask[:, :, 0] # squeezed mask of shape (n, m)
