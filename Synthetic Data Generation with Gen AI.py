#!/usr/bin/env python
# coding: utf-8

# <b><h1><span style="color:Orange">Synthetic Data Generation using GAN's</span></h1></b> 

# To get started with the task of Synthetic Data Generation, we need a dataset that we can use to feed into a Generative Adversarial Networks (GANs) model, which will be trained to generate new data samples that will be similar to the original data and the relationships between the features in the original data.

# The dataset contains daily records of insights into app usage patterns over time. The goal will be to generate synthetic data that mimics the original dataset by ensuring that it maintains the same statistical properties while providing privacy for users actual usage behaviour.

# In[1]:


import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('C:\\Users\\hp\\Downloads\\screentime_analysis.csv')

data.head()


# In[12]:


data.info() #Column summary


# Date: The date of the screentime data.
# Usage: Total usage time of the app (likely in minutes).
# Notifications: The number of notifications received.
# Times opened: The number of times the app was opened.
# App: The name of the app

# # Data Preprocessing 
# 

# In[4]:


# drop unnecessary columns
data_gan = data.drop(columns=['Date', 'App'])

# initialize a MinMaxScaler to normalize the data between 0 and 1
scaler = MinMaxScaler()

# normalize the data
normalized_data = scaler.fit_transform(data_gan)

# convert back to a DataFrame
normalized_df = pd.DataFrame(normalized_data, columns=data_gan.columns)

normalized_df.head()


# In[5]:


#The generator will take a latent noise vector as input and generate a synthetic sample similar to the data.


# In[6]:


latent_dim = 100  # latent space dimension (size of the random noise input vector)

def build_generator(latent_dim):
    model = Sequential([
        Dense(128, input_dim=latent_dim),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(256),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(512),
        LeakyReLU(alpha=0.01),
        BatchNormalization(momentum=0.8),
        Dense(3, activation='sigmoid')  # output layer for generating 3 features
    ])
    return model

#create the generator
generator = build_generator(latent_dim)
generator.summary()


# Now, we will build a discriminator that will take a real or synthetic data sample and classify it as real or fake

# In[14]:


def build_discriminator():
    model = Sequential([
        Dense(512, input_shape=(3,)),
        LeakyReLU(alpha=0.01),
        Dense(256),
        LeakyReLU(alpha=0.01),
        Dense(128),
        LeakyReLU(alpha=0.01),
        Dense(1, activation='sigmoid')  # output: 1 neuron for real/fake classification
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

#create the discriminator

discriminator = build_discriminator()
discriminator.summary()


# In[9]:


#Now, we will freeze the discriminator’s weights when training the generator to ensure only the generator is updated during those training steps.This is crucial so that there is no critic from the discriminator during the training 


# In[10]:


def build_gan(generator, discriminator):
    # freeze the discriminator’s weights while training the generator
    discriminator.trainable = False

    model = Sequential([generator, discriminator])
    model.compile(loss='binary_crossentropy', optimizer=Adam())
    return model

# create the GAN
gan = build_gan(generator, discriminator)
gan.summary()


# Now, we will train the GAN using the following steps:
# 
# 1.Generate random noise.
# 2.Use the generator to create fake data.
# 3.Train the discriminator on both real and fake data.
# 4.Train the generator via the GAN to fool the discriminator.

# In[11]:


def train_gan(gan, generator, discriminator, data, epochs=10000, batch_size=128, latent_dim=100):
    for epoch in range(epochs):
        # select a random batch of real data
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]

        # generate a batch of fake data
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_data = generator.predict(noise)

        # labels for real and fake data
        real_labels = np.ones((batch_size, 1))  # real data has label 1
        fake_labels = np.zeros((batch_size, 1))  # fake data has label 0

        # train the discriminator
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        # train the generator via the GAN
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_labels = np.ones((batch_size, 1)) #labelled as 1 so discriminator classifies it as real
        g_loss = gan.train_on_batch(noise, valid_labels)

        # print the progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: D Loss: {0.5 * np.add(d_loss_real, d_loss_fake)}, G Loss: {g_loss}")

train_gan(gan, generator, discriminator, normalized_data, epochs=10000, batch_size=128, latent_dim=latent_dim)


# #### Now, here’s how we can use the generator to create new synthetic data:

# In[13]:


# generate new data
noise = np.random.normal(0, 1, (1000, latent_dim))  # generate 1000 synthetic samples
generated_data = generator.predict(noise)

# convert the generated data back to the original scale
generated_data_rescaled = scaler.inverse_transform(generated_data)

# convert to DataFrame
generated_df = pd.DataFrame(generated_data_rescaled, columns=data_gan.columns)

generated_df.head()


# <b><h3><span style="color:Green">In this project, we explored the task of synthetic data generation with Generative AI using Generative Adversarial Networks (GANs). We started by preprocessing a dataset of app usage insights by focusing on features like Usage, Notifications, and Times opened which were normalized for GAN training. The GAN architecture was built with a generator to create synthetic data and a discriminator to distinguish between real and generated data</span></h3></b> 
