import torch
from torchvision import models, transforms
import cv2
import sys
import numpy as np
from tqdm import tqdm

def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad

def save(mask, img, blurred):
    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    
    heatmap = np.float32(heatmap) / 255
    cam = 1.0*heatmap + np.float32(img)/255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)	

    cv2.imwrite("perturbated.png", np.uint8(255*perturbated))
    cv2.imwrite("heatmap.png", np.uint8(255*heatmap))
    cv2.imwrite("mask.png", np.uint8(255*mask))
    cv2.imwrite("cam.png", np.uint8(255*cam))

def perturb(image, model, transforms, device='gpu', \
    tv_beta=3, lr=0.1, max_iter=500, l1_coeff=0.01, tv_coeff=0.2):
    original_img = cv2.imread(image)
    original_img = cv2.resize(original_img, (224, 224))
    img = np.float32(original_img) / 255
    blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
    blurred_img2 = np.float32(cv2.medianBlur(original_img, 11))/255
    blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
    mask = torch.ones((28, 28), dtype = torch.float32).unsqueeze(0).unsqueeze(0)

    img = transforms(img).unsqueeze(0)
    blurred = transforms(blurred_img2)

    if (device == 'cpu'):
        upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
    optimizer = torch.optim.Adam([mask], lr=lr)

    target = torch.nn.Softmax(dim=1)(model(img))  # debug
    category = np.argmax(target.cpu().data.numpy())
    print( "Category with highest probability: ", category)

    for i in tqdm(range(max_iter)):
        #print(mask.shape)
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image, 
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                        upsampled_mask.size(3))
        
        # Use the mask to perturbated the input image.
        perturbated_input = img.mul(upsampled_mask) + \
                            blurred.mul(1-upsampled_mask)
        
        noise = np.zeros((224, 224, 3), dtype = np.float32)
        cv2.randn(noise, 0, 0.2)
        noise = np.transpose(noise, (2, 0, 1))
        noise = torch.from_numpy(noise).unsqueeze(0)  # debug
        #print(noise.shape)
        perturbated_input = perturbated_input + noise
        
        outputs = torch.nn.Softmax(dim=1)(model(perturbated_input))  # debug
        loss = l1_coeff*torch.mean(torch.abs(1 - mask)) + \
                tv_coeff*tv_norm(mask, tv_beta) + outputs[0, category]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)

    upsampled_mask = upsample(mask)
    save(upsampled_mask, original_img, blurred_img_numpy)