# Explaining Class-Specific Activation Mappings with Meaningful Perturbations

![](images/collage.png?raw=true)

This repository is a Pytorch implementation of the algorithm described in [Interpretable Explanations of Black Boxes by Meaningful Perturbation (Fong, et. al., 2018)](https://arxiv.org/pdf/1704.03296.pdf). This  is meant to be applied to Image-based classifiers.

## Usage

This method can work with black-box neural networks as a _post-hoc_ explanation of a classifier output. Simply import the function perturb in any of your existing projects like this:

```python
from perturb import perturb

perturb(image, your_model, output_folder, transforms, plot=True)
```

An example implementation of both perturbation for interpretability and de-noising adversarial images on the VIT Transformer can be seen in the Jupyter notebook. 

## How it works

![](images/eq.PNG?raw=true)

Minimal Deletion: The algorithm creates a mask for the regions of the image that the model essentially used to predict its label. Since the model is dependent on these regions to make its prediction, masking these regions causes the model to change its prediction. 

By minimizing the regions that are masked, we can identify the most essential features of the image according to the model AND quantify how much those regions affect the model's prediction by measuring the change in the model's confidence on the original label before and after the mask is applied. This accounts for the 1st and 3rd terms.


Smooth Perturbations: Sometimes, creating a mask via SGD leads to  exploiting the artifacts found in the model. It therefore creates masks that do not correspond to any semantically-observable features (e.x. noise, wings, eyes) found in the image, which hinders their interpretability. The authors therefore add noise to the mask so that the explainer does not learn from a single mask (3rd term) and Total Variation (TV) norm (2nd term). This leads to a smooth mask.

## Adversarial Noise Detection

The paper also reports that their method creates a markedly different mask if perturbing an image with adversarial noise compared to a mask created to perturb a clean image.

![](images/adversarial.png?raw=true)