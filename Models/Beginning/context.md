# Beginning

Here, I will attempt to gather basic information about the most important models involved today in image/video generation, and provide important informationa nd context of the field/how these models came about.

> Although understanding state-of-the-art models is the goal, I beleive it is also important to understand older models, in order to understand the context/thinking behind why the knew ones where developed.


# Image Generation Evolution

Here are some of the first models that preceeded Diffusion models

## 0. Early Attempts
- Attempts to generate images using AI were done for many years, but with little progress.
- However, soon after the popularization of deep networks with many layers (CNNs like alexnet), something called a *Generative Adversarial Network* was developed (GAN)

## 1. GANs
[GANs by iangoodfellow](https://arxiv.org/pdf/1406.2661)
- Developed by Ian Goddfellow in 2014 at Univ. Montreal
    1. 'Generative Model' that attempts to model the distribution of the training set
    2. Model that tries to predict whether the some data comes from the generative model, or the actual training set
- These two different neural nets basically are 'competing' to try to trick (generative) or predict (predictor) eachother.

- The end result (ideally) should be that the generative model can produce images that are indistinguishable from real data, fooling a trained predictor

- Can be used for *many* tasks, such as creating fake music that is indestinguishable from real music, or generating images from sketches

### Limitations of GANS
- Unstable trainings due to the dual-network model: the generative model can start producing only a few images and learn the 'beat' the discrimitor in a 'bad' way that is not alligned with creating new, original images. Therefore, prone to overfitting/memorizing training data as well
- Require a lot of compute
- Hard to manipulate desired output (only really representative of training set)

> Adversarial: networks working against eachother!

## 2. VAEs (variational auto-encoders)
- Autoencoder that takes high-dimensional variable and encodes it.
- Key Idea:
    1. This 'clustering' mechanism is good for classfication tasks, where we want an encoder to classify a digit based on its pixels.
    2. However, if we want to produce dynamic images, like a dog with a crown, we wouldn't 'average' dog and crown! Therefore, we need some type of continous space to model 
    3. For example, we can 'reconstruct' samples ranging from 7 to 1


## 3. Transformers role in Image generation
- Incorporating transformers into GANs

## 4. Diffusion


Examples of models:
## 1. DALLE-E V1 (OpenAI)
> [paper](https://arxiv.org/pdf/2102.12092)

- Utilized LLM/transformer architecture with VAE's

## 5. Diffusion