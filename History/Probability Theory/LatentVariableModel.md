# Latent variable models

> VAE's are a type of latent-variable model, and we will be using this probabilistic perspective throughout our generative AI journey!

### Intro: AutoEncoders:
Learn to encode and reconstruct an image with a 'bottleneck', which is our latent-vector/latent-space variables

- We we train our autoencoder, our Neural net is able to deconstruct and reconstruct images, however when we try to sample new images from our 'latent distribution', we get blurred and odd pixels (our space is unstructured)

- We want  need to guide our model to be aware of the dependencies between 'similar' images/classes of images

- So, ***how do we make our latent spacae mean something?*** If we could capture complex semantic relationships in our space (IE: move right for happy, left for sad, up for guy, down for girl), we could get one step closer to generating images

>
> Below is an illustration of what we hope or latent space would encode vs. what a naive one would:
![latentspaces](latentspacecomp.png)

We can see how having a meaningful latent space in an autoencoder would allow us to generate meaningful images if we sampled from the space.

--------------------------------------------------------------------------

# Probabilistic approach w/ latent variable models

----------------------------------------------------------

## Entrop and KL divergence:

- Entropy is a concept related to the field of information theory, and is an important way that we can describe the uncertainty or 'surprise' of systems

- We will use this idea to quantify how well our model works, and for other things as well later on

- Read this out for a more full understanding:

![Entropy-Through-KL](entropy-through-kl.png)

-------------------------------------------------------------------------------------

## Defining our problem

sources: 
- [princeton](https://pillowlab.princeton.edu/teaching/statneuro2018/slides/notes16_LatentsAndEM.pdf)

- [stanford](https://ermongroup.github.io/cs228-notes/learning/latent/)

So, we are able to define our KL divergence for Q(x), an approximation of P(x) with our understanding of KL divergence.

Let's go from defining our problem with a theoretical probability approach, and looking towards machine learning to solve our problems.

- We observe *D* from our continous high-dimensional distribution *X* (pixel PDF)
- We have some latent variables that are high-level (continuous) descriptions of X such as emotional content

1. We cannot sample from our probability distribution *X* of all possible images (x has p(x) probability of being real in *X*)
2. We therefore need a stable (gaussian), lower-dimensional representation of *X*,
which is P(Z)
3. Now we only need to find P(Z|X), or the mapping of X onto Z such that the probability reflects the real probability or P(X)

If we observe *D* we could use bayes theorem:

    P(Z|X=D) = P(X=D|Z)P(Z)/(P(X=D))

    where P(X=D) is ∫...∫ P(Z|X)P(Z)dz...dz_n 

    this marginalization is intractable (cannot be solved)

So, we need a surrogate: *Q(Z)*, where *Q* is an approximation of P(Z), which is the actual latent-distribution of X.

    Q(Z) ~ p(Z|X=D)
    Our distribution Q(Z) is approximating P(Z|X=D) under the observed data *D*

-----------------

### Sources:
- [VAE-math-explained](https://www.youtube.com/watch?v=iwEzwTTalbg&ab_channel=UmarJamil)
- [Wikipedia-Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory))
- [KL-Divergence-Explained](https://www.youtube.com/watch?v=KHVR587oW8I&ab_channel=ArtemKirsanov)