---
title: "Facial Keypoint Detection with Neural Networks"
layout: post
date: 2022-04-12 22:10
tag: CNN
#image: https://sergiokopplin.github.io/indigo/assets/images/jekyll-logo-light-solid.png
#headerImage: true
projects: true
hidden: true # don't count this post in blog pagination
description: "Using neural networks for facial keypoint detection."
category: project
author: jasonding
externalLink: false
---

# Facial Keypoint Detection with Neural Networks

Acknowledgement: This project is from CS194-26 Project 5 at UC Berkeley. More info can be found [here](https://inst.eecs.berkeley.edu/~cs194-26/fa21/hw/proj5/).

In this project, I will use convolutional neural networks to automatically detect facial keypoints. The tool I use is PyTorch.

## Part 1: Nose Tip Detection

In this part, I will detect only the nose tip point.

### Dataloader

To preprocess the images, I turn them into grayscale and then normalize them to scale from -0.5 to 0.5. The output of `skimage.color.rgb2gray` already gives floating values from 0 to 1. All I have to do is just deduct 0.5 off. The images are resized to (60, 80). Then, I need to define the nose tip dataset. It inherits PyTorch's Dataset class. For the `__getitem()__` function, I use the example code for help. The dataloader wraps up this dataset with `batch_size=1` since it is a small dataset.

Below are some sampled images visualized with ground-truth keypoints. 

<center class="only">    <img src="./result/samples.png" width="400"/>    <figcaption>samples with nose tip</figcaption> </center>

### CNN

The model I use is below.

<center class="only">    <img src="./result/model_nose.png" width="600"/>    <figcaption>model for nose tip detection</figcaption> </center>

### **Loss Function and Optimizer**

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
```

I use MSE(mean squared error loss) as loss function and Adam with learning rate 1e-3 as optimizer.

I split my dataset to 192 images of training set and 48 images of validation set. I train for 25 epochs, and below is the the train and validation MSE loss plot across epochs.

<center class="only">    <img src="./result/nose_mse.jpg" width="600"/>    <figcaption>train and validation MSE loss</figcaption> </center>

The curves are not as smooth as I think, but it shows a good trend. Also, the validation loss still fluctuates. I will improve this in the next section.

### **Hyperparameter Tuning**

The first parameter I modify is the learning rate. I decrement it to 1e-4. The plot is below.

<center class="only">    <img src="./result/nose_mse2.jpg" width="600"/>    <figcaption>train and validation MSE loss</figcaption> </center>

The curves become much more smooth. However, the loss is not very good.

Another parameter I modify is the channel number. I decrease it. Below is the new structure of my model.

<center class="only">    <img src="./result/model_nose_mod.png" width="600"/>    <figcaption>modified model</figcaption> </center>

And also the plot.

<center class="only">    <img src="./result/nose_mse3.jpg" width="600"/>    <figcaption>train and validation MSE loss</figcaption> </center>

You can see that it is the worst among all three. It tells us that we should use more channels.

### Prediction

Green points are the ground-truth points, while the red ones are my predictions.

Below are two facial images which my model detects the nose correctly.

<center class="half">    <img src="./result/nose_1.jpg" width="400"/>    <figcaption>id=1</figcaption><img src="./result/nose_209.jpg" width="400"/><figcaption>id=209</figcaption> </center>

And 2 more images where it detects incorrectly.

<center class="half">    <img src="./result/nose_9.jpg" width="400"/>    <figcaption>id=9</figcaption><img src="./result/nose_10.jpg" width="400"/><figcaption>id=10</figcaption> </center>

I think it fails because the man/woman is not facing to the front. Instead, he/she changes the posture, which makes my simple network hard to detect it correctly.



## Part 2: Full Facial Keypoints Detection

Now, in this part, I will detect all 58 landmarks instead of just the nose tip.

### Dataloader

The process is very similar to part 1, but one difference is that all the images are resized to (120, 160). In addition, I add data augmentation to prevent my model from overfitting. I use rotation and ColorJitter. My dataloader uses `batch_size=4`. The images are still split in the same way as part 1.

Below are some sample images along with transformations.

<center class="only">    <img src="./result/samples_full.png" width="400"/>    <figcaption>samples with all landmarks</figcaption> </center>

### CNN

The model is below.

<center class="only">    <img src="./result/model_full.png" width="600"/>    <figcaption>model full</figcaption> </center>

### Training

The loss function and optimizer I use is below.

```python
full_criterion = nn.MSELoss()
full_optimizer = optim.Adam(fullnet.parameters(), lr=5e-5)
```

I train for 25 epochs. Below is the plot showing both training and validation loss across iterations. Again, training set has the first 192 images and validation set has the rest.

<center class="only">    <img src="./result/full_mse.jpg" width="600"/>    <figcaption>train and validation MSE loss</figcaption> </center>

The plot doesn't show the loss clearly. The actual values are below.

<center class="only">    <img src="./result/mse_list.png" width="250"/>    <figcaption>train and validation MSE loss</figcaption> </center>

### Prediction

Green points are the ground-truth points, while the red ones are my predictions.

Below are two facial images which my model detects the landmarks correctly.

<center class="half">    <img src="./result/full_1.jpg" width="400"/>    <figcaption>id=1</figcaption><img src="./result/full_208.jpg" width="400"/><figcaption>id=208</figcaption> </center>

And 2 more images where it detects incorrectly.

<center class="half">    <img src="./result/full_209.jpg" width="400"/>    <figcaption>id=209</figcaption><img src="./result/full_213.jpg" width="400"/><figcaption>id=213</figcaption> </center>

It fails because the model is not too adapted to the random transformations and the posture is very differernt than others (id=213, the man is facing sideways, which is a very different posture) and my model doesn't learn that well.

### Learned filters

Below are the 12 filters of my first convolution layer.

<center class="six">    <img src="./filters/filter_0.jpg" width="200"/>    <figcaption>filter 0</figcaption> <img src="./filters/filter_1.jpg" width="200"/>    <figcaption>filter 1</figcaption> <img src="./filters/filter_2.jpg" width="200"/>    <figcaption>filter 2</figcaption> <img src="./filters/filter_3.jpg" width="200"/>    <figcaption>filter 3</figcaption> <img src="./filters/filter_4.jpg" width="200"/>    <figcaption>filter 4</figcaption> <img src="./filters/filter_5.jpg" width="200"/>    <figcaption>filter 5</figcaption>  </center>

<center class="six">    <img src="./filters/filter_6.jpg" width="200"/>    <figcaption>filter 6</figcaption> <img src="./filters/filter_7.jpg" width="200"/>    <figcaption>filter 7</figcaption> <img src="./filters/filter_8.jpg" width="200"/>    <figcaption>filter 8</figcaption> <img src="./filters/filter_9.jpg" width="200"/>    <figcaption>filter 9</figcaption> <img src="./filters/filter_10.jpg" width="200"/>    <figcaption>filter 10</figcaption> <img src="./filters/filter_11.jpg" width="200"/>    <figcaption>filter 11</figcaption>  </center>

Unfortunately, I can't find any human-readable information from these filters.



## Part 3: Train With Larger Dataset

For this part, I will use a larger dataset(ibug) for training a facial keypoints detector. This dataset contains 6666 images of varying image sizes, and each image has 68 annotated facial keypoints. 

### Dataloader

I use the example code to script the lanmarks and bounding boxes of each image. For bounding boxes with negative values, I simply skip those images. I crop the image by that bounding box, and resize it to (224, 224) in grayscale. Like the previous two parts, I use (0,1) ratio of the image as my landmarks instead of the actual coordinates. The bounding boxes are not very accurate, so I scale the width and height by 1.5. For data augmentation, I use Gaussian Blur, adding Linear Contrast, Gaussian Noise, changing Brightness, and Affine Transformation(scaling, translation, rotation). This is achieved with the help of `imgaug` package.

### CNN

I use ResNet18 as my model. I have two modifications: 1. input channel of the first convolution layer is set to 1 instead of 3; 2. output size of the last fully connected layer is set to 136 instead of 1000 to predict the 68 landmarks.

Below is the detailed structure.

<center class="half">    <img src="./result/resnet1.png" width="400"/>    <figcaption>ResNet18</figcaption><img src="./result/resnet2.png" width="400"/><figcaption>ResNet18(con't)</figcaption> </center>

### Training

The loss function and optimizer I use is below.

```python
criterion = nn.MSELoss()
optimizer = optim.Adam(newmodel.parameters(), lr=1e-4, weight_decay=3e-5)
```

For training, I random split the dataset and set up training and validation dataloder as below.

```python
train_set, val_set = torch.utils.data.random_split(dataset, [5952, 578], generator=torch.Generator().manual_seed(23))
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4, worker_init_fn = worker_init_fn)
test_dataloader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=4, worker_init_fn = worker_init_fn)
```

I train for 20 epochs. Below is the plot showing both training and validation loss across iterations.

<center class="only">    <img src="./result/resnet_mse.jpg" width="600"/>    <figcaption>Training and Validation MSE</figcaption> </center>

From there, I decide to choose the previous hyperparameters and use the entire dataset to train my model.  I train for 30 epochs. The full dataset training MSE is below.

<center class="only">    <img src="./result/resnet_mse_full.jpg" width="600"/>    <figcaption>Full Dataset MSE</figcaption> </center>

### Prediction

For the test set, I create a new dataset with no transformation. The images are resized to (224, 224) and to grayscale, but for the final submission, they are converted back according to the instructions.

Below are some predictions from the test set. Notice that I don't have the ground-truth landmarks because they are from the test set.

<center class="half">    <img src="./result/res_success.png" width="400"/>    <figcaption>Two Success Cases</figcaption><img src="./result/res_fail.png" width="400"/><figcaption>Two Failure Cases</figcaption> </center>

I think some detections fail because my model is not trained too well for different face shapes.

After doing some research, I decide to use a **new model** for detecting the landmarks, and it is exactly what **Bells & Whistles** ask me to do!



## >Bells & Whistles: Part1

The key takeaway is to turn the regression problem of predicting the keypoint coordinates into a pixelwise classification problem. After introducing upsampling via transpose convolution, I can have 68 heatmaps each corresponding to one facial landmark.

### Heatmap

The first step is to create heatmaps from landmarks using 2D Gaussian distribution at that keypoint location. Below is the visualization (I choose sigma=5).

<center class="half">    <img src="./result/img_0.jpg" width="300"/>    <figcaption>image</figcaption><img src="./result/gauss_0.jpg" width="500"/><figcaption>68 gaussian dists</figcaption> </center>

<center class="half">    <img src="./result/img_1.jpg" width="300"/>    <figcaption>image</figcaption><img src="./result/gauss_1.jpg" width="500"/><figcaption>68 gaussian dists</figcaption> </center>

Also some single landmark gaussian distributions.

<center class="four">    <img src="./result/gauss_single_30.jpg" width="300"/>    <figcaption>landmark 30</figcaption><img src="./result/gauss_single_31.jpg" width="300"/><figcaption>landmark 31</figcaption><img src="./result/gauss_single_32.jpg" width="300"/>    <figcaption>landmark 32</figcaption><img src="./result/gauss_single_33.jpg" width="300"/><figcaption>landmark 33</figcaption> </center>

### Model

Luckily, I can use the pretrained FCN_ResNet50 model with modifications: 1. the input channel of the backbone conv1 is set to 1 instead of 3; 2. the output of last classifier layer is set to 136 instead of 21 and also the auxiliary classifier. The structure is below.

<center class="five">    <img src="./result/fcn_1.png" width="400"/>    <figcaption>FCN_ResNet50_1</figcaption><img src="./result/fcn_2.png" width="400"/>    <figcaption>FCN_ResNet50_2</figcaption><img src="./result/fcn_3.png" width="400"/>    <figcaption>FCN_ResNet50_3</figcaption><img src="./result/fcn_4.png" width="400"/>    <figcaption>FCN_ResNet50_4</figcaption></figcaption><img src="./result/fcn_5.png" width="400"/>    <figcaption>FCN_ResNet50_5</figcaption> </center>

### Training

I use the same metrics and split, and I train for 10 epochs. Below is the plot.

<center class="only">    <img src="./result/fcn_mse.jpg" width="600"/>    <figcaption>Training and Validation MSE</figcaption> </center>

From there, I decide to choose the previous hyperparameters and use the entire dataset to train my model.   For this training only, I **use** **all the** **6666** **images** to get a better result on Kaggle. As I mentioned earlier, some of the bounding boxes are not correct. If the coordinate of the top left corner of the box contains negative value, I set it to 0 with no harm. I train for 50 epochs. The full dataset training MSE is below.

<center class="only">    <img src="./result/fcn_mse_full.jpg" width="600"/>    <figcaption>Full MSE</figcaption> </center>

There are some fluctuations, but the overall trend is showing that my model is converging.

### Back to Coords

To transform the heatmaps back to the (x,y) coordinates, I use the weighted average of the top-n points (density) of the heatmap as the keypoint. I use n=25. However, this method is not perfect. I may spend more time on choosing that n compared with MSE. Also, if there is a keypoint around the corner, then I should use less top-n points, but if it's in the center, then I should use more. I didn't implement this method, but I will if I have enough time.

### Prediction

Here is one output of the model (68, 224, 224).

<center class="half">    <img src="./result/gauss_pred.jpg" width="400"/>    <figcaption>Output of the model</figcaption><img src="./result/pred.jpg" width="400"/><figcaption>Convert back to the coordinates</figcaption> </center>

Comparing to the previous prediction by ResNet18, it is much better!

Here are some **test** images with the predicted landmarks on the original images.

<center class="three">    <img src="./result/pred_0.jpg" width="500"/>    <figcaption>id=0</figcaption><img src="./result/pred_2.jpg" width="500"/><figcaption>id=2</figcaption><img src="./result/pred_3.jpg" width="500"/><figcaption>id=3</figcaption> </center>

### Kaggle

<center class="only">    <img src="./result/kaggle.png" width="800"/>    <figcaption>Kaggle Score</figcaption> </center>

### My own photos

<center class="three">    <img src="./result/myself_pt.jpg" width="500"/>    <figcaption>Me</figcaption><img src="./result/chris_evans_pt.jpg" width="500"/><figcaption>Chris Evans</figcaption><img src="./result/mu_li_pt.jpg" width="500"/><figcaption>Mu Li</figcaption> </center>

The detection looks pretty good! I think my new model is now very good at detecting facial landmarks. And for my own photos, all three people have standard postures, which makes my model good at detecting the landmarks.

## Takeaways

1. If you try to flip the image as data augmentation, don't forget to change the landmarks' order! God knows how much time I spent on this.
2. If you enlarge the bounding boxes, the actual size of the training image may not be the same as the size of the bounding boxes (box['left']+box['width'] may exceed the image size).
3. Relating keypoints detection to segmentation is interesting and powerful.
4. Training models is tough, but the result is meaningful.



## References

1. PyTorch tutorial. https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
2. How to use FCN on keypoints. https://fairyonice.github.io/Achieving-top-5-in-Kaggles-facial-keypoints-detection-using-FCN.html
3. Chris Evans. https://en.wikipedia.org/wiki/Chris_Evans_(actor)
4. Mu Li. https://www.zhihu.com/people/mli65

