# EchoGANhance

## Premise

Cardiac ultrasound (echocardiography) is a cornerstone of the diagnostic and treatment algorithm for many cardiac diseases and is cheaper, safer, and more comfortable for the patient than alternative cardiovascular imaging modalities. Unfortunately, while echocardiography has remarkable temporal and spatial resolution, image quality can suffer from a low signal-to-noise ratio and a variety of artifacts that are produced by natural imaging barriers. The purpose of this project is to train a deep generative model that can produce high quality, diagnostic echocardiography images from lower quality studies that suffer from these artifacts. 

![Echocardiogram](Ultrasound_of_human_heart_apical_4-cahmber_view.gif)

![Poor image quality](poor-imaghe-stress-11_30102012.gif)

The purpose of this project is to train a deep generative model for unpaired image to image translation that can produce high quality, diagnostic echocardiography images from lower quality studies that suffer from these artifacts. 

CycleGANs
The cycleGAN (Zhu et al.) model allows for unpaired image to image translation by training two separate generators (GA->B and GB->A) and two separate discriminators (DA and DB) (see Figure 1). Each image from each domain is first converted to the opposite domain, then converted back to the original domain using the appropriate generators. This cyclic generation ensures that the transformations preserve the general features of the source image by enforcing a cycle consistency loss between the cyclically generated image and the original input image.  Moreover, an identity mapping loss serves as a sort of regularizer of the generators, by ensuring that if the GA->B is used for an image from the B domain, the same image is produced (identity) and vice versa. 

 ![CycleGANs](https://user-images.githubusercontent.com/65331476/121422726-21a76400-c935-11eb-99bd-f8f8a4d2b243.png)


 ![image](https://user-images.githubusercontent.com/65331476/121422817-3683f780-c935-11eb-8c16-80320b3e8bc3.png)

CycleGANs are by nature susceptible to “hallucinating” features that are not in the original input image, particularly if there is over/underrepresentation of certain features in the training set since they are based on distribution matching losses (Cohen et al.). This can have critical implications in a domain where high-fidelity translation is essential, such as the medical imaging domain.

### UNetGAN

![image](https://user-images.githubusercontent.com/65331476/121423684-26b8e300-c936-11eb-9e89-852b0108fc46.png)

### PatchGAN

![image](https://user-images.githubusercontent.com/65331476/121423640-1acd2100-c936-11eb-8036-755aec489118.png)

![Overall Loss Function](https://user-images.githubusercontent.com/65331476/121423381-cfb30e00-c935-11eb-85a8-10ace3ab99b1.png)

## Dataset

## Training

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/rmw362/CS496ProjectWebsite/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
