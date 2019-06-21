---
title: 学习使用TensorFlow来识别交通标志
date: 2018-09-13 10:50:14
tags:
- tensorflow
categories:
- tensorflow
comments: true
---

# 前言

本文参考https://juejin.im/entry/5a1637f2f265da432528f6ef  的文章和  https://github.com/waleedka/traffic-signs-tensorflow  的源代码。   

 给定交通标志的图像，我们的模型应该能够知道它的类型。  
 首先我们要导入需要的库。


```python
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import transform
import random
```

    /home/song/anaconda3/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters

   <!-- more -->
## 1  加载数据和分析数据

### 1.1 加载数据

我们使用的是Belgian Traffic Sign Dataset。网址为http://btsd.ethz.ch/shareddata/  
在这个网站可以下载到我们需要的数据集。你只需要下载BelgiumTS for Classification (cropped images):后面的两个数据集：  
    
    BelgiumTSC_Training (171.3MBytes)  
    BelgiumTSC_Testing (76.5MBytes)  
  我把这两个数据集分别放在了以下的路径：    
    
    /home/song/Downloads/BelgiumTSC_Training/Training  
    /home/song/Downloads/BelgiumTSC_Testing/Testing  
  Training目录包含具有从00000到00061的序列号的子目录。目录名称表示从0到61的标签，每个目录中的图像表示属于该标签的交通标志。 图像以不常见的.ppm格式保存，但幸运的是，这种格式在skimage库中得到了支持。


```python
def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                  if os.path.isdir(os.path.join(data_dir, d))]

    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH = "/home/song/Downloads/"
train_data_dir = os.path.join(ROOT_PATH, "BelgiumTSC_Training/Training")
test_data_dir = os.path.join(ROOT_PATH, "BelgiumTSC_Testing/Testing")

images, labels = load_data(train_data_dir)
```

### 1.2 分析数据

我们可以看一下我们的训练集中有多少图片和标签：


```python
print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))
```

    Unique Labels: 62
    Total Images: 4575

这里的set很有意思，可以看一下这篇文章：http://www.voidcn.com/article/p-uekeyeby-hn.html  
这里的set很有意思，可以看一下这篇文章：http://www.voidcn.com/article/p-uekeyeby-hn.html  
在处理一系列数据时，如果需要剔除重复项，则通常采用set数据类型。本身labels里面是有很多重复的元素的，但set(labels)就剔除了重复项。可以通过print(labels)和print(set(labels))命令查看一下两者输出的有什么区别。  
我们还可以通过画直方图来看一下数据的分布情况。


```python
plt.hist(labels,62)
plt.show()
```


![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/58270023.jpg)


可以看出，该数据集中有的标签的分量比其它标签更重：标签 22、32、38 和 61 显然出类拔萃。这一点之后我们会更深入地了解。

### 1.3 可视化数据

#### 1.3.1 热身

我们可以先随机地选取几个交通标志将其显示出来。我们还可以看一下图片的尺寸。我们还可以看一下图片的最小值和最大值，这是验证数据范围并及早发现错误的一个简单方法。其中的plt.axis('off')是为了不在图片上显示坐标尺，大家可以注释掉这句话看看如果去掉有什么不一样。


```python
traffic_signs=[100,1050,3650,4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    #plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape, 
                                                  images[traffic_signs[i]].min(), 
                                                  images[traffic_signs[i]].max()))
```


![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/91631184.jpg)


    shape: (292, 290, 3), min: 0, max: 255
    

![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/94366993.jpg)


    shape: (132, 139, 3), min: 4, max: 255
    

![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/55563405.jpg)


    shape: (146, 110, 3), min: 7, max: 255
    

![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/66096551.jpg)


    shape: (110, 105, 3), min: 3, max: 255

大多数神经网络需要固定大小的输入，我们的网络也不例外。 但正如我们上面所看到的，我们的图像大小并不完全相同。 一种常见的方法是将图像裁剪并填充到选定的纵横比，但是我们必须确保在这个过程中我们不会切断部分交通标志。 这似乎需要进行手动操作！ 我们其实有一个更简单的解决方案，即我们将图像大小调整为固定大小，并忽略由不同长宽比导致的失真。 这时，即使图片被压缩或拉伸了一点，我们也可以很容易地识别交通标志。我们用下面的命令将图片的尺寸调整为32*32。
大多数神经网络需要固定大小的输入，我们的网络也不例外。 但正如我们上面所看到的，我们的图像大小并不完全相同。 一种常见的方法是将图像裁剪并填充到选定的纵横比，但是我们必须确保在这个过程中我们不会切断部分交通标志。 这似乎需要进行手动操作！ 我们其实有一个更简单的解决方案，即我们将图像大小调整为固定大小，并忽略由不同长宽比导致的失真。 这时，即使图片被压缩或拉伸了一点，我们也可以很容易地识别交通标志。我们用下面的命令将图片的尺寸调整为32*32。

#### 1.3.2 重调图片的大小


```python
images32 = [transform.resize(image,(32,32)) for image in images]
```

    /home/song/.local/lib/python3.6/site-packages/skimage/transform/_warps.py:105: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    /home/song/.local/lib/python3.6/site-packages/skimage/transform/_warps.py:110: UserWarning: Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.
      warn("Anti-aliasing will be enabled by default in skimage 0.15 to "

重新运行上面随机显示交通标志的代码。
重新运行上面随机显示交通标志的代码。


```python
traffic_signs=[100,1050,3650,4000]

for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images32[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images32[traffic_signs[i]].shape, 
                                                  images32[traffic_signs[i]].min(), 
                                                  images32[traffic_signs[i]].max()))
```


![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/65434369.jpg)


    shape: (32, 32, 3), min: 0.0, max: 1.0
    

![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/59269206.jpg)


    shape: (32, 32, 3), min: 0.038373161764705975, max: 1.0
    

![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/62497379.jpg)


    shape: (32, 32, 3), min: 0.05559895833333348, max: 1.0
    

![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/90866008.jpg)


    shape: (32, 32, 3), min: 0.048665364583333495, max: 1.0

从上面的图和shape的值都能看出，图片的尺寸一样大了。最小值和最大值现在的范围在0和1.0之间，和我们未调整图片大小时的范围不同。
从上面的图和shape的值都能看出，图片的尺寸一样大了。最小值和最大值现在的范围在0和1.0之间，和我们未调整图片大小时的范围不同。

#### 1.3.3 显示每一个标签下的第一张图片

之前我们在直方图中看过62个标签的分布情况。现在我们尝试将每个标签下的第一张图片显示出来，另外还可以通过列表的count()方法来统计某个标签出现的次数，也就是能统计出有多少张图片对应该标签。我们可以定义一个函数，名为display_images_and_labels，你当然可以定义成别的名字，不过定义函数是为了之后可以方便地调用。以下分别显示出了未调整尺寸和已调整尺寸的交通标志图。


```python
def display_images_and_labels(images, labels):
    """Display the first image of each label."""
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        plt.imshow(image)
    

display_images_and_labels(images, labels)
display_images_and_labels(images32, labels)


```


![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/19586062.jpg)



![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/15441432.jpg)


正如我们在直方图中看到的那样，具有标签 22、32、38 和 61 的交通标志要明显多得多。图中可以看到标签 22 有 375 个实例，标签 32 有 316 实例，标签 38 有 285 个实例，标签 61 有 282 个实例。

#### 1.3.4 显示某一个标签下的交通标志

看过每个标签下的第一张图片之后，我们可以将某一个标签下的图片展开显示出来，看看这个标签下的是否是同一类交通标志。我们不需要把该标签下的所有图片都显示出来，可以只展示24张，你可以更改为其他的数字，显示更多或者更少。我们这里选择标签为21的看一下，在之前的图片中可以看到，label 21对应于stop标志。


```python
def display_label_images(images, label):
    """Display images of a specific label."""
    limit = 24  # show a max of 24 images
    plt.figure(figsize=(15, 5))
    i = 1

    start = labels.index(label)
    end = start + labels.count(label)
    for image in images[start:end][:limit]:
        plt.subplot(3, 8, i)  # 3 rows, 8 per row
        plt.axis('off')
        i += 1
        plt.imshow(image)


display_label_images(images32,21)
```


![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/52152061.jpg)


可以看出，label 21对应的前24张图片都是stop标志。不难推测，整个label 21对应的应都是stop标志。

## 2 构建深度网络 

### 2.1 构建TensorFlow图并训练

首先，我们创建一个Graph对象。TensorFlow有一个默认的全局图，但是我们不建议使用它。设置全局变量通常太容易引入错误了，因此我们自己创建一个图。之后设置占位符来放图片和标签。注意这里参数x的维度是 [None, 32, 32, 3]，这四个参数分别表示 [批量大小，高度，宽度，通道] （通常缩写为 NHWC）。我们定义了一个全连接层，并使用了relu激活函数进行非线性操作。我们通过argmax()函数找到logits最大值对应的索引，也就是预测的标签了。之后定义loss函数，并选择合适的优化算法。这里选择Adam算法，因为它的收敛速度比一般的梯度下降算法更快。这个时候我们只刚刚构建图，并且描述了输入。我们定义的变量，比如，loss和predicted_labels，它们都不包含具体的数值。它们是我们接下来要执行的操作的引用。我们要创建会话才能开始训练。我这里把循环次数设置为301，并且如果i是10的倍数，就打印loss的值。


```python
g = tf.Graph()

with g.as_default():
    # Initialize placeholders 
    x = tf.placeholder(dtype = tf.float32, shape = [None, 32, 32,3])
    y = tf.placeholder(dtype = tf.int32, shape = [None])

    # Flatten the input data
    images_flat = tf.contrib.layers.flatten(x)
    #print(images_flat)
    
    # Fully connected layer 
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

     # Convert logits to label 
    predicted_labels = tf.argmax(logits, 1)
    
    # Define a loss function
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                logits = logits))

    # Define an optimizer 
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    print("images_flat: ", images_flat)
    print("logits: ", logits)
    print("loss: ", loss)
    print("predicted_labels: ", predicted_labels)

    sess=tf.Session(graph=g)
    sess.run(tf.global_variables_initializer())
    for i in range(301):
        #print('EPOCH', i)
        _,loss_value = sess.run([train_op, loss], feed_dict={x: images32, y: labels}) 
        if i % 10 == 0:
            print("Loss: ", loss_value)
        #print('DONE WITH EPOCH')
```

    images_flat:  Tensor("Flatten/flatten/Reshape:0", shape=(?, 3072), dtype=float32)
    logits:  Tensor("fully_connected/Relu:0", shape=(?, 62), dtype=float32)
    loss:  Tensor("Mean:0", shape=(), dtype=float32)
    predicted_labels:  Tensor("ArgMax:0", shape=(?,), dtype=int64)
    Loss:  4.181018
    Loss:  3.0714655
    Loss:  2.6622696
    Loss:  2.4586942
    Loss:  2.3419585
    Loss:  2.2633858
    Loss:  2.2044215
    Loss:  2.157206
    Loss:  2.1180305
    Loss:  2.0847433
    Loss:  2.0559382
    Loss:  2.030667
    Loss:  2.008251
    Loss:  1.9882014
    Loss:  1.9701369
    Loss:  1.9537587
    Loss:  1.938837
    Loss:  1.9251733
    Loss:  1.912607
    Loss:  1.9010073
    Loss:  1.8902632
    Loss:  1.8802778
    Loss:  1.8709714
    Loss:  1.8622767
    Loss:  1.8541412
    Loss:  1.8465083
    Loss:  1.8393359
    Loss:  1.8325756
    Loss:  1.8261962
    Loss:  1.8201678
    Loss:  1.8144621

### 2.2使用模型
### 2.2使用模型

现在我们用sess.run()来使用我们训练好的模型，并随机取了训练集中的10个图片进行分类，并同时打印了真实的标签结果和预测结果。


```python
# Pick 10 random images
sample_indexes = random.sample(range(len(images32)), 10)
sample_images = [images32[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "predicted_labels" op.
predicted = sess.run([predicted_labels], 
                        feed_dict={x: sample_images})[0]
print(sample_labels)
print(predicted)
```

    [41, 39, 1, 53, 21, 22, 38, 48, 7, 53]
    [41 39  1 53 21 22 40 47  7 53]
    

```python
​```python
fig=plt.figure(figsize=(10,10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5,2,1+i)
    plt.axis("off")
    color='green' if truth == prediction else 'red'
    plt.text(40,10,"Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_images[i])
```


![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/94071068.jpg)


### 2.3评估模型

以上，我们的模型只在训练集上是可以正常运行的，但是它对于其他的未知数据集的泛化能力如何呢？我们可以在测试集当中进行评估。我们还可以计算一下准确率。


```python
test_images, test_labels = load_data(test_data_dir)
test_images32 = [transform.resize(image, (32, 32))
                 for image in test_images]
display_images_and_labels(test_images32, test_labels)

# Calculate how many matches we got.
predicted = sess.run([predicted_labels], 
                        feed_dict={x: test_images32})[0]
match_count = sum([int(y == y_) 
                   for y, y_ in zip(test_labels, predicted)])
accuracy = match_count / len(test_labels)
print("Accuracy: {:.4f}".format(accuracy))



# Pick 10 random images
sample_test_indexes = random.sample(range(len(test_images32)), 10)
sample_test_images = [test_images32[i] for i in sample_test_indexes]
sample_test_labels = [test_labels[i] for i in sample_test_indexes]

# Run the "predicted_labels" op.
test_predicted = sess.run([predicted_labels], 
                        feed_dict={x: sample_test_images})[0]
print(sample_test_labels)
print(test_predicted)

fig=plt.figure(figsize=(10,10))
for i in range(len(sample_test_images)):
    truth = sample_test_labels[i]
    prediction = test_predicted[i]
    plt.subplot(5,2,1+i)
    plt.axis("off")
    color='green' if truth == prediction else 'red'
    plt.text(40,10,"Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
             fontsize=12, color=color)
    plt.imshow(sample_test_images[i])
```

    /home/song/.local/lib/python3.6/site-packages/skimage/transform/_warps.py:105: UserWarning: The default mode, 'constant', will be changed to 'reflect' in skimage 0.15.
      warn("The default mode, 'constant', will be changed to 'reflect' in "
    /home/song/.local/lib/python3.6/site-packages/skimage/transform/_warps.py:110: UserWarning: Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts when down-sampling images.
      warn("Anti-aliasing will be enabled by default in skimage 0.15 to "
    
    
    Accuracy: 0.5631
    [38, 35, 19, 32, 32, 7, 13, 38, 18, 38]
    [39  0 19 32 32  7 13 40 17 39]
    

![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/67359700.jpg)

![png](http://boketuchuang.oss-cn-beijing.aliyuncs.com/18-9-13/73349594.jpg)


### 2.4关闭会话


```python
sess.close()
```

最后，记得关闭会话。
