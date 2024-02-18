# Image-Captioning
Image Captioning Show, Attend and Tell (SAT)
The purpose of Image Captioning Show, Attend and Tell (SAT) is to automatically generate descriptions of images using a novel attention mechanism.
# Generate human-like captions: 
Unlike previous models that relied solely on image features, SAT aimed to create captions that resembled natural language descriptions humans would produce. This included capturing details, objects, relationships, and even actions within the image.

# This introduces an attention based image caption generator. 
# The model changes its attention to the relevant part of the image while it generates each word.

# Objective
To build a model that can generate a descriptive caption for an image we provide it.

In the interest of keeping things simple, let's implement the Show, Attend, and Tell paper. This is by no means the current state-of-the-art, but is still pretty darn amazing. 
This model learns where to look.

As you generate a caption, word by word, you can see the model's gaze shifting across the image.

This is possible because of its Attention mechanism, which allows it to focus on the part of the image most relevant to the word it is going to utter next.

# Concepts

Image captioning. duh.

**Encoder-Decoder architecture**. Typically, a model that generates sequences will use an Encoder to encode the input into a fixed form and a Decoder to decode it, word by word, into a sequence.

**Attention**. The use of Attention networks is widespread in deep learning, and with good reason. This is a way for a model to choose only those parts of the encoding that it thinks is relevant to the task at hand. The same mechanism you see employed here can be used in any model where the Encoder's output has multiple points in space or time. In image captioning, you consider some pixels more important than others. In sequence to sequence tasks like machine translation, you consider some words more important than others.

**Transfer Learning**. This is when you borrow from an existing model by using parts of it in a new model. This is almost always better than training a new model from scratch (i.e., knowing nothing). As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand. Using pretrained word embeddings is a dumb but valid example. For our image captioning problem, we will use a pretrained Encoder, and then fine-tune it as needed.

# Encoder
The Encoder encodes the input image with 3 color channels into a smaller image with "learned" channels.

This smaller encoded image is a summary representation of all that's useful in the original image.

Since we want to encode images, we use Convolutional Neural Networks (CNNs).

We don't need to train an encoder from scratch. Why? Because there are already CNNs trained to represent images.

For years, people have been building models that are extraordinarily good at classifying an image into one of a thousand categories. It stands to reason that these models capture the essence of an image very well.

I have chosen to use the 101 layered Residual Network trained on the ImageNet classification task, already available in PyTorch. As stated earlier, this is an example of Transfer Learning. You have the option of fine-tuning it to improve performance.

![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/be3e9306-42d2-4aca-ad40-24f29e1eef08)

These models progressively create smaller and smaller representations of the original image, and each subsequent representation is more "learned", with a greater number of channels. The final encoding produced by our ResNet-101 encoder has a size of 14x14 with 2048 channels, i.e., a 2048, 14, 14 size tensor.

I encourage you to experiment with other pre-trained architectures. The paper uses a VGGnet, also pretrained on ImageNet, but without fine-tuning. Either way, modifications are necessary. Since the last layer or two of these models are linear layers coupled with softmax activation for classification, we strip them away.

# Decoder
The Decoder's job is to look at the encoded image and generate a caption word by word.

Since it's generating a sequence, it would need to be a Recurrent Neural Network (RNN). We will use an LSTM.

In a typical setting without Attention, you could simply average the encoded image across all pixels. You could then feed this, with or without a linear transformation, into the Decoder as its first hidden state and generate the caption. Each predicted word is used to generate the next word.
![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/bd493949-96a4-4880-aa1d-55f8174e288d)

In a setting with Attention, we want the Decoder to be able to look at different parts of the image at different points in the sequence. For example, while generating the word football in a man holds a football, the Decoder would know to focus on – you guessed it – the football!

![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/94c753de-1e6d-430d-a4a4-136a8ee3329b)

Instead of the simple average, we use the weighted average across all pixels, with the weights of the important pixels being greater. This weighted representation of the image can be concatenated with the previously generated word at each step to generate the next word.

# Attention
The Attention network computes these weights.

Intuitively, how would you estimate the importance of a certain part of an image? You would need to be aware of the sequence you have generated so far, so you can look at the image and decide what needs describing next. For example, after you mention *a man*, it is logical to declare that he is *holding a football*.

This is exactly what the Attention mechanism does – it considers the sequence generated thus far, and attends to the part of the image that needs describing next.
![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/f628bf45-c7e3-441d-b10c-33d5c75421ba)

We will use soft Attention, where the weights of the pixels add up to 1. If there are P pixels in our encoded image, then at each timestep t –

![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/98e3ef31-b762-47cf-82ef-3c2fb7a97996)

You could interpret this entire process as computing the probability that a pixel is the place to look to generate the next word.

# Putting it all together
It might be clear by now what our combined network looks like.

![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/db4db281-2f8b-4419-8034-b4f831f136ae)

![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/8ae94866-8853-4ba9-b6c3-9a619f8955dd)


Once the Encoder generates the encoded image, we transform the encoding to create the initial hidden state h (and cell state C) for the LSTM Decoder.
At each decode step,
the encoded image and the previous hidden state is used to generate weights for each pixel in the Attention network.
the previously generated word and the weighted average of the encoding are fed to the LSTM Decoder to generate the next word.

# Results

![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/ffeb4e04-fed8-4ecf-8a82-2836e4fd62d9)
![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/e31bef28-a038-426b-937c-c0021e54195b)
![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/d20316ee-27a3-412d-a402-954a9928edee)
![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/9fffa5c7-2f21-417e-ad46-616cc1848cd6)
![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/4d2357ab-587e-4afd-8a1c-992b91c6ef5d)
![image](https://github.com/MARCUS-MITc/Image-Captioning/assets/123622512/6d259dc3-1b65-4106-87a6-7b28d2238f3b)











