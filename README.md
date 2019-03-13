# Udacity ML





## 数据集大全

- UCI数据集：https://archive.ics.uci.edu/ml/datasets.html





## 额外资料

##### 审阅老师的笔记：

- [机器学习笔记](https://github.com/LeanderLXZ/machine-learning-notes)



#####Adaboost：

 * 来自 Schapire 很棒的[教程](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/schapire.pdf)
 * 这是一篇由 Freund 和 Schapire 合著的原始[论文](https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf)
 * 由 Freund 和 Schapire 合著的关于Adaboost几项实验的后续[论文](https://people.cs.pitt.edu/~milos/courses/cs2750/Readings/boosting.pdf)

#####层次聚类

- [Using Hierarchical Clustering of Secreted Protein Families to Classify and Rank Candidate Effectors of Rust Fungi](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0029847)



##### DBSCAN

- [Anomaly detection in temperature data using dbscan algorithm](https://ieeexplore.ieee.org/abstract/document/5946052/)
- [Traffic Classification Using Clustering Algorithms](https://pages.cpsc.ucalgary.ca/~mahanti/papers/clustering.pdf)



##### ICA

 - [独立成分分析：算法与应用](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/10.1.1.322.679.pdf)



##### PCA

- [参考资料：Interpretation of the Principal Components](https://onlinecourses.science.psu.edu/stat505/node/54)



##### 神经网络

- 我们的内容开发者之一 Jay Alammar 创建了这个神奇的神经网络 “游乐场”，在这里你可以看到很棒的可视化效果，并可以使用参数来解决线性回归问题，然后尝试一些神经网络回归。 预祝学习愉快！

  [https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/](http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)



##### 深度神经网络(MLP)

- [卷积神经网络](https://baike.baidu.com/item/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/17541100?fr=aladdin)

- Keras 中有很多优化程序，建议你访问此[链接](https://keras.io/optimizers/)或这篇精彩[博文](http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)（此链接来自外网，国内网络可能打不开），详细了解这些优化程序。这些优化程序结合使用了上述技巧，以及其他一些技巧。最常见的包括：

  #### SGD

  这是随机梯度下降。它使用了以下参数：

  - 学习速率。
  - 动量（获取前几步的加权平均值，以便获得动量而不至于陷在局部最低点）。
  - Nesterov 动量（当最接近解决方案时，它会减缓梯度）。

  #### Adam

  Adam (Adaptive Moment Estimation) 使用更复杂的指数衰减，不仅仅会考虑平均值（第一个动量），并且会考虑前几步的方差（第二个动量）。

  #### RMSProp

  RMSProp (RMS 表示均方根误差）通过除以按指数衰减的平方梯度均值来减小学习速率。

- 我们的内容开发者之一 Jay Alammar 创建了这个神奇的神经网络 “游乐场”，在这里你可以看到很棒的可视化效果，并可以使用参数来解决线性回归问题，然后尝试一些神经网络回归。 预祝学习愉快！

  [https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/](http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)



##### 卷积神经网络(CNN)

- 了解 [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/) 模型。

  - 如果你能训练人工智能机器人唱歌，干嘛还训练它聊天？在 2017 年 4 月，研究人员使用 WaveNet 模型的变体生成了歌曲。原始论文和演示可以在[此处](http://www.creativeai.net/posts/W2C3baXvf2yJSLbY6/a-neural-parametric-singing-synthesizer)找到。

- 了解[文本分类 CNN](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)。

  - 你或许想注册作者的[深度学习简讯](https://www.getrevue.co/profile/wildml)！

- 了解 Facebook 的[创新 CNN 方法](https://code.facebook.com/posts/1978007565818999/a-novel-approach-to-neural-machine-translation/)(Facebook)，该方法专门用于解决语言翻译任务，准确率达到了前沿性水平，并且速度是 RNN 模型的 9 倍。

- 利用 CNN 和强化学习玩 [Atari 游戏](https://deepmind.com/research/dqn/)。你可以[下载](https://sites.google.com/a/deepmind.com/dqn/)此论文附带的代码。

  - 如果你想研究一些（深度强化学习）初学者代码，建议你参阅 Andrej Karpathy 的[帖子](http://karpathy.github.io/2016/05/31/rl/)。

- 利用 CNN 玩[看图说词游戏](https://quickdraw.withgoogle.com/#)！

  - 此外，还可以参阅 [A.I.Experiments](https://aiexperiments.withgoogle.com/) 网站上的所有其他很酷的实现。别忘了 [AutoDraw](https://www.autodraw.com/)！

- 详细了解 [AlphaGo](https://deepmind.com/research/alphago/)。

  - 阅读[这篇文章](https://www.technologyreview.com/s/604273/finding-solace-in-defeat-by-artificial-intelligence/?set=604287)，其中提出了一个问题：如果掌控 Go“需要人类直觉”，那么人性受到挑战是什么感觉？_

- 观看这些非常酷的视频，其中的无人机都受到 CNN 的支持。
  - 这是初创企业 [Intelligent Flying Machines (IFM)](https://www.youtube.com/watch?v=AMDiR61f86Y) (Youtube)的访谈。
  - 户外自主导航通常都要借助[全球定位系统 (GPS)](http://www.droneomega.com/gps-drone-navigation-works/)，但是下面的演示展示的是由 CNN 提供技术支持的[自主无人机](https://www.youtube.com/watch?v=wSFYOw4VIYY)(Youtube)。

- 如果你对无人驾驶汽车使用的 CNN 感兴趣，请参阅：
  - 我们的[无人驾驶汽车工程师纳米学位课程](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)，我们在[此项目](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project)中对[德国交通标志](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)数据集中的标志进行分类。
  - 我们的[机器学习工程师纳米学位课程](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009)，我们在[此项目](https://github.com/udacity/machine-learning/tree/master/projects/digit_recognition)中对[街景门牌号](http://ufldl.stanford.edu/housenumbers/)数据集中的门牌号进行分类。
  - 这些[系列博客](https://pythonprogramming.net/game-frames-open-cv-python-plays-gta-v/)，其中详细讲述了如何训练用 Python 编写的 CNN，以便生成能够玩“侠盗猎车手”的无人驾驶 AI。

- 参阅视频中没有提到的其他应用情形。
  - 一些全球最著名的画作被[转换成了三维形式](http://www.businessinsider.com/3d-printed-works-of-art-for-the-blind-2016-1)，以便视力受损人士也能欣赏。虽然这篇文章没有提到是怎么做到的，我们注意到可以使用 CNN [预测单个图片的深度](https://www.cs.nyu.edu/~deigen/depth/)。
  - 参阅这篇关于使用 CNN 确定乳腺癌位置的[研究论文](https://research.googleblog.com/2017/03/assisting-pathologists-in-detecting.html)(google research)。
  - CNN 被用来[拯救濒危物种](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)！
  - 一款叫做 [FaceApp](http://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/) 的应用使用 CNN 让你在照片中是微笑状态或改变性别。

- **损失函数**是用来估量模型中预测值y与真实值Y之间的差异，即不一致程度

- 如果你想详细了解 Keras 中的完全连接层，请阅读这篇关于密集层的[文档](https://keras.io/layers/core/)。你可以通过为 `kernel_initializer` 和 `bias_initializer` 参数提供值更改权重的初始化方法。注意默认值分别为 `'glorot_uniform'` 和 `'zeros'`。你可以在相应的 Keras [文档](https://keras.io/initializers/)中详细了解每种初始化程序的工作方法。

- Keras 中有很多不同的[损失函数](https://keras.io/losses/)。对于这节课来说，我们将仅使用 `categorical_crossentropy`。

- 参阅 Keras 中

  可用的优化程序列表

  。当你编译模型（在记事本的第 7 步）时就会指定优化程序。

  - `'sgd'` : SGD
  - `'rmsprop'` : RMSprop
  - `'adagrad'` : Adagrad
  - `'adadelta'` : Adadelta
  - `'adam'` : Adam
  - `'adamax'` : Adamax
  - `'nadam'` : Nadam
  - `'tfoptimizer'` : TFOptimize

- 在训练过程中，你可以使用很多回调（例如 ModelCheckpoint）来监控你的模型。你可以参阅此处的[详情内容](https://keras.io/callbacks/#modelcheckpoint)。建议你先详细了解 EarlyStopping 回调。如果你想查看另一个 ModelCheckpoint 代码示例，请参阅[这篇博文](http://machinelearningmastery.com/check-point-deep-learning-models-keras/)。

- 请参阅该 Keras [文档](https://keras.io/layers/pooling/)，了解不同类型的池化层！;论文[network in network](https://arxiv.org/abs/1312.4400)

- 这是用于在 Keras 中指定神经网络（包括 CNN）的[备忘单](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Keras_Cheat_Sheet_Python.pdf)。

- 参阅 CIFAR-10 竞赛的[获胜架构](http://blog.kaggle.com/2015/01/02/cifar-10-competition-winners-interviews-with-dr-ben-graham-phil-culliton-zygmunt-zajac/)！

- 阅读这篇对 MNIST 数据集进行可视化的[精彩博文](http://machinelearningmastery.com/image-augmentation-deep-learning-keras/)。

- 参阅此[详细实现](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)，了解如何使用增强功能提高 Kaggle 数据集的效果。

- 阅读关于 ImageDataGenerator 类的 Keras [文档](https://keras.io/preprocessing/image/)。

- 参阅 [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) 论文！

- 在此处详细了解 [VGGNet](https://arxiv.org/pdf/1409.1556.pdf)。

- 此处是 [ResNet](https://arxiv.org/pdf/1512.03385v1.pdf) 论文。

- 这是用于访问一些著名 CNN 架构的 Keras [文档](https://keras.io/applications/)。

- 阅读这一关于梯度消失问题的[详细处理方案](http://neuralnetworksanddeeplearning.com/chap5.html)。

- 这是包含不同 CNN 架构的基准的 GitHub [资源库](https://github.com/jcjohnson/cnn-benchmarks)。

- 访问 [ImageNet Large Scale Visual Recognition Competition (ILSVRC)](http://www.image-net.org/challenges/LSVRC/) 网站。

- 详细了解如何解读 CNN（尤其是卷积层），建议查看以下资料：

  - 这是摘自斯坦福大学的 CS231n 课程中的一个a [章节](http://cs231n.github.io/understanding-cnn/)，其中对 CNN 学习的内容进行了可视化。
  - 参阅这个关于很酷的 [OpenFrameworks](http://openframeworks.cc/) 应用的[演示](https://aiexperiments.withgoogle.com/what-neural-nets-see)，该应用可以根据用户提供的视频实时可视化 CNN！
  - 这是另一个 CNN 可视化工具的[演示](https://www.youtube.com/watch?v=AgkfIQ4IGaM&t=78s)。如果你想详细了解这些可视化图表是如何制作的，请观看此[视频](https://www.youtube.com/watch?v=ghEmQSxT6tw&t=5s)。
  - 这是另一个可与 Keras 和 Tensorflow 中的 CNN 无缝合作的[可视化工具](https://medium.com/merantix/picasso-a-free-open-source-visualizer-for-cnns-d8ed3a35cfc5)。
  - 阅读这篇可视化 CNN 如何看待这个世界的 [Keras 博文](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)。在此博文中，你会找到 Deep Dreams 的简单介绍，以及在 Keras 中自己编写 Deep Dreams 的代码。阅读了这篇博文后：
    - 再观看这个利用 Deep Dreams 的[音乐视频](https://www.youtube.com/watch?v=XatXy6ZhKZw)（注意 3:15-3:40 部分）！
    - 使用这个[网站](https://deepdreamgenerator.com/)创建自己的 Deep Dreams（不用编写任何代码！）。
  - 如果你想详细了解 CNN 的解释
    - 这篇[文章](https://blog.openai.com/adversarial-example-research/)详细讲解了在现实生活中使用深度学习模型（暂时无法解释）的一些危险性。
    - 这一领域有很多热点研究。[这些作者](https://arxiv.org/abs/1611.03530)最近朝着正确的方向迈出了一步。

- 参阅这篇 [研究论文](https://arxiv.org/pdf/1411.1792.pdf)，该论文系统地分析了预先训练过的 CNN 中的特征的可迁移性。

- 阅读这篇详细介绍 Sebastian Thrun 的癌症检测 CNN 的[《自然》论文](http://www.nature.com/articles/nature21056.epdf?referrer_access_token=_snzJ5POVSgpHutcNN4lEtRgN0jAjWel9jnR3ZoTv0NXpMHRAJy8Qn10ys2O4tuP9jVts1q2g1KBbk3Pd3AelZ36FalmvJLxw1ypYW0UxU7iShiMp86DmQ5Sh3wOBhXDm9idRXzicpVoBBhnUsXHzVUdYCPiVV0Slqf-Q25Ntb1SX_HAv3aFVSRgPbogozIHYQE3zSkyIghcAppAjrIkw1HtSwMvZ1PXrt6fVYXt-dvwXKEtdCN8qEHg0vbfl4_m&tracking_referrer=edition.cnn.com)！

- 这是提议将 GAP 层级用于对象定位的[首篇研究论文](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)。

- 参阅这个使用 CNN 进行对象定位的[资源库](https://github.com/alexisbcook/ResNetCAM-keras)。

- 观看这个关于使用 CNN 进行对象定位的[视频演示](https://www.youtube.com/watch?v=fZvOy0VXWAI)(Youtube链接，国内网络可能打不开)。

- 参阅这个使用可视化机器更好地理解瓶颈特征的[资源库](https://github.com/alexisbcook/keras_transfer_cifar10)。

- ImageNet 这目前一个非常流行的数据集，常被用来测试图像分类等计算机视觉任务相关的算法。它包含超过一千万个 URL，每一个都链接到 [1000 categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) 中所对应的一个物体的图像。任给输入一个图像，该 ResNet-50 模型会返回一个对图像中物体的预测结果

- 推荐你阅读以下材料来加深对 CNN和Transfer Learning的理解:

  - [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
  - [Using Convolutional Neural Networks to Classify Dog Breeds](http://cs231n.stanford.edu/reports/2015/pdfs/fcdh_FinalReport.pdf)
  - [Building an Image Classifier](https://towardsdatascience.com/learning-about-data-science-building-an-image-classifier-part-2-a7bcc6d5e825)
  - [Tips/Tricks in CNN](http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html)
  - [Transfer Learning using Keras](https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8)
  - [Transfer Learning in TensorFlow on the Kaggle Rainforest competition](https://medium.com/@luckylwk/transfer-learning-in-tensorflow-on-the-kaggle-rainforest-competition-4e978fadb571)
  - [Transfer Learning and Fine-tuning](https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2)
  - [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
  - [简述迁移学习在深度学习中的应用](https://www.jiqizhixin.com/articles/2018-01-04-7)
  - [无需数学背景，读懂 ResNet、Inception 和 Xception 三大变革性架构](https://www.jiqizhixin.com/articles/2017-08-19-4)

- [[VGG16\] VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION](https://arxiv.org/abs/1409.1556)

- [[Inception-v1\] Going deeper with convolutions](https://arxiv.org/abs/1409.4842)

- [[Inception-v3\] Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

- [[Inception-v4\] Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

- [[ResNet\] Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

- [[Xception\] Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

- Haar

  - [Tutorial - Face Detection using Haar Cascades](https://docs.opencv.org/3.3.0/d7/d8b/tutorial_py_face_detection.html)
  - [Face Detection using OpenCV](https://www.superdatascience.com/opencv-face-detection/)
  - [OpenCV Face Detection in Images using Haar Cascades with Face Count](https://shahsparx.me/opencv-face-detection-haar-cascades/)
  - [YouTube video - Haar Cascade Object Detection Face & Eye](https://www.youtube.com/watch?v=88HdqNDQsEk)
  - [Haar caascade classifiers](http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Object_Detection_Face_Detection_Haar_Cascade_Classifiers.php)
  - [YouTube video - VIOLA JONES FACE DETECTION EXPLAINED](https://www.youtube.com/watch?v=_QZLbR67fUU)
  - [How can I understand Haar-like feature for face detection?](https://www.quora.com/How-can-I-understand-Haar-like-feature-for-face-detection)
  - [A simple facial recognition api for Python and the command line](https://github.com/ageitgey/face_recognition)
  - [这个知乎专栏](https://zhuanlan.zhihu.com/p/24816781)介绍了目前主流的基于深度学习的人脸识别算法。

- 改进模型的一些思路：

  1. **交叉验证（Cross Validation）**
     在本次训练中，我们只进行了一次训练集/测试集切分，而在实际模型训练过程中，我们往往是使用交叉验证（Cross Validation）来进行模型选择（Model Selection）和调参（Parameter Tunning）的。交叉验证的通常做法是，按照某种方式多次进行训练集/测试集切分，最终取平均值（加权平均值），具体可以参考[维基百科](https://en.wikipedia.org/wiki/Cross-validation_(statistics))的介绍。
  2. **模型融合/集成学习（Model Ensembling）**
     通过利用一些机器学习中模型融合的技术，如voting、bagging、blending以及staking等，可以显著提高模型的准确率与鲁棒性，且几乎没有风险。你可以参考我整理的机器学习笔记中的[Ensemble部分](https://github.com/LeanderLXZ/machine-learning-notes#step5)。
  3. **更多的数据**
     对于深度学习（机器学习）任务来说，更多的数据意味着更为丰富的输入空间，可以带来更好的训练效果。我们可以通过数据增强（Data Augmentation）、[对抗生成网络（Generative Adversarial Networks）](https://www.ams.giti.waseda.ac.jp/data/pdf-files/2017_IEVC_watabe.pdf)等方式来对数据集进行扩充，同时这种方式也能提升模型的鲁棒性。
  4. **更换人脸检测算法**
     尽管OpenCV工具包非常方便并且高效，Haar级联检测也是一个可以直接使用的强力算法，但是这些算法仍然不能获得很高的准确率，并且需要用户提供正面照片，这带来的一定的不便。所以如果想要获得更好的用户体验和准确率，我们可以尝试一些新的人脸识别算法，如基于深度学习的一些算法。
  5. **多目标监测**
     更进一步，我们可以通过一些先进的目标识别算法，如RCNN、Fast-RCNN、Faster-RCNN或Masked-RCNN等，来完成一张照片中同时出现多个目标的检测任务。

##### 强化学习

- [教科书](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)

  

## 数据预处理

##### 定义异常值的法则

 - [ Tukey 的定义异常值的方法](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/)



##### 统计量含义

- [数据的统计量特征](http://blog.csdn.net/trierwang/article/details/4855309)





## 模型评价方法

##### 资料

 - [模型评价标准大全（英文)](https://blog.exsilio.com/all/accuracy-precision-recall-f1-score-interpretation-of-performance-measures/)



##### 正则化对比

 	![](C:\Users\Barnett\Desktop\Udacity笔记\img\L1和L2正则化对比.png)

- L1：使用L1正则化时，我们希望得到稀疏向量，它表示较小权重趋向于0.所以你如果想降低权重值，最终得到较小的数，有利于模型选择
- L2：不支持稀疏向量，因为他确保所有权重一致较小，这样一般可以训练模型，得到更好的结果

#### 模型复杂度图表

![](.\img\模型复杂度图表.png)

- 说明：

  ![](.\img\模型复杂度图表example.png)





##### 混淆矩阵

​	

|          | Guessed Positive      | Guessed Negative      |
| -------- | --------------------- | --------------------- |
| Positive | True Positives（TP）  | False Negatives（FN） |
| Negative | False Positives（FP） | True Negatives（TN）  |

##### Accuracy(准确率)

​	$Accuracy = \frac{TP+TN}{TP+FN+FP+TN}$



##### Precision（精度/查准率：真阳性在阳性中的占比）

​	$Precision=\frac{TP}{TP+FP}$



##### Recall（召回率/查全率：真阳性在猜测中的正确的占比）

​	$Recall=\frac{TP}{TP+FN}$



##### F score

​	$F-Beta-score​$.

​	$F_\beta=(1+\beta^2) * \frac{precision*recall}{\beta^2*precision+recall}​$

​	同时考虑查准率和查全率，当$\beta=0.5$时，更多强调的是，记为$F_\beta - score$



> $注意，在 F_\beta*F**β* 得分公式中，如果设为 \beta = 0,*β*=0, 则$
>
> $F_0 = (1+0^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{0 \cdot \text{Precision} + \text{Recall}} = \frac{\text{Precision} \cdot \text{Recall}}{\text{Recall}} = \text{Precision}因此， \beta的最低值为 0，这时候就得出精度。$
>
> 
>
> $注意，如果 N 非常大，则​$
>
> $F_\beta = (1+N^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{N^2 \cdot \text{Precision} + \text{Recall}} = \frac{\text{Precision} \cdot \text{Recall}}{\frac{N^2}{1+N^2}\text{Precision} + \frac{1}{1+N^2}\text{Recall}}​$
>
> $随着 N变成无穷大，可以看出 \frac{1}{1+N^2}变成 0，并且 \frac{N^2}{1+N^2} 变成 1.$
>
> 因此，如果取极限值，则
>
> ${\lim_{N\rightarrow \infty}} F_N = \frac{\text{Precision} \cdot \text{Recall}}{1 \cdot \text{Precision} + 0 \cdot \text{Recall}} = \text{Recall}$
>
> - 如果$ \beta = 0​$则得出**精度**。
> - 如果 $\beta = \infty​$则得出**召回率**。
> - 对于其他$\beta$值，如果接近 0，则得出接近精度的值，如果很大，则得出接近召回率的值，如果$ \beta = 1$则得出精度和召回率的**调和平均数**。



##### ROC曲线(受试者工作特性曲线)

​	即是不同分割下的TP/AP-FP/AP曲线  -- AP: ALL POSITIVE



##### 回归指标/决定系数（sklearn: r2_score）

​	$R2 = 1 - \frac{mean -square- error- of -model}{mean- square- error -of -random -model}​$

​	

​	坏模型：模型的均方误差和随机模型的均方误差相近，R2结果接近于0

​	好模型：模型的均方误差大于随机模型的均方误差，R2结果接近于1



##### 聚类验证

- 外部指标

  如果数据集有分类标签，则使用外部指标

  | Index               | Range   | Available in sklearn | More                                                         |
  | ------------------- | ------- | -------------------- | ------------------------------------------------------------ |
  | Adjusted Rand Score | [-1, 1] | Yes                  | - [Details of the Adjusted Rand index](http://faculty.washington.edu/kayee/pca/supp.pdf) |
  | Fawlks and mallows  | [0, 1]  | Yes                  |                                                              |
  | NMI measure         | [0, 1]  | Yes                  |                                                              |
  | Jaccard             | [0, 1]  | Yes                  |                                                              |
  | F-measure           | [0, 1]  | Yes                  |                                                              |
  | Purity              | [0, 1]  | No                   |                                                              |

  

- 内部指标

  数据集没有分类标签

  | Index             | Range   | Available in sklearn | More |
  | ----------------- | ------- | -------------------- | ---- |
  | Silhouette Index  | [-1, 1] | Yes                  |      |
  | Calinski-Harabasz |         | Yes                  |      |
  | BIC               |         |                      |      |
  | Dunn Index        |         |                      |      |

  - Silhouette Index(轮廓系数)

    $S_i = \frac{b_i-a_i}{max(a_i, b_i)}$

    -- $a是同一个聚类中到其他样本的平均距离$

    -- $b是与它距离最近不同聚类中到样本的平均距离$

    

    $S = average(S_1, S_2, ...,S_n)$

    

    **注意，使用DBSCAN不能使用该指标，该指标没有噪声的概念，连接论文有介绍DBSCAN的指标方法[基于密度的聚类验证](https://s3.cn-north-1.amazonaws.com.cn/static-documents/nd101/MLND+documents/10.1.1.707.9034.pdf)**

    

    

- 相对指标

  - 紧凑性
  - 可分性





## 聚类分析过程

![](.\img\聚类分析过程.png)

- 数据集
- 特征提取（对数据进行转换，以生成新的有用特征）/软聚类
- 聚类算法选择
- 聚类验证
- 结果解释
- 知识







## 感知器算法



##### 模型

![](.\img\感知器模型.png)





#####算法流程：

感知器步骤如下所示。对于坐标轴为$ (p,q) $的点，标签 y，以及等式 $\hat{y}$=step($w_{1}x_{1}+w_{2}x_{2}+b$) 给出的预测

- 如果点分类正确，则什么也不做。

- 如果点分类为正，但是标签为负，则分别减去 $\alpha p, \alpha q,αp,αq​$, 和 $\alpha​$ 至 $w_1, w_2,w_1,w_2,​$和 $b​$。(**即高于分类线的错误分类点减更新**)

- 如果点分类为负，但是标签为正，则分别将 $\alpha p, \alpha q,αp,αq, 和 \alpha$ 加到 $w_1, w_2,w_1,w_2,和 b 上。$(**即低于于分类线的错误分类点加更新**)

  ​

## 决策树算法



#####熵的计算一般公式（只有两个类别，数量分别为$m, n$）：

​	$Entropy=\frac{m}{m-n}log_2(\frac{m}{m+m})-\frac{n}{m+n}log_2(\frac{n}{m+n})$

#####推广公式：

​	$H=-\sum\limits_{i=1}^np(x_i)log_2p(x_i)$  ------ $n$是分类的数目,$p(x_i)$是某个类别出现的概率



#####条件熵：

​	$H(Y|X)=\sum\limits_{i=1}^np_iH(Y|X=x_i)​$

​	其中 $p_i=P(X=x_i), i=1, 2, \cdots, n​$



#####信息增益：

​	分类前的熵和分类后的熵均值的差，为信息增益。

​	$g(D, A) = H(D) - H(D|A)$

#####信息增益比：

​	特征$A$对训练数据集$D$的信息增益$g_R(D,A)$，定义为其信息增益$g(D,A)$与训练集D的经验熵$H(D)$之比。

​	$g_R(D,A)=g(D,A)/H(D)$

#####信息增益最大化

​	分类前后信息增益最大的分类方法



#####决策树的超参数

- 最大深度（max_depth）
- 每片叶子的最小样本数（min_samples_leaf）
- 每次分裂的最小样本数（类似上一个参数）（min_samples_split）
- 最大特征数（max_features）



#####ID3算法

> ID3算法就是用信息增益大小来判断当前节点应该用什么特征来构建决策树，用计算出的信息增益最大的特征来建立决策树的当前节点。
>
> **不足：**
>
> ID3算法虽然提出了新思路，但是还是有很多值得改进的地方。　　
>
> a)ID3没有考虑连续特征，比如长度，密度都是连续值，无法在ID3运用。这大大限制了ID3的用途。
>
> b)ID3采用信息增益大的特征优先建立决策树的节点。很快就被人发现，在相同条件下，取值比较多的特征比取值少的特征信息增益大。比如一个变量有2个值，各为1/2，另一个变量为3个值，各为1/3，其实他们都是完全不确定的变量，但是取3个值的比取2个值的信息增益大。如果校正这个问题呢？
>
> c) ID3算法对于缺失值的情况没有做考虑
>
> d) 没有考虑过拟合的问题
>



#####C4.5算法（ID3的改良）

> 对于**第一个问题**，不能处理连续特征， C4.5的思路是将连续的特征离散化。 
> 对于**第二个问题**，信息增益作为标准容易偏向于取值较多的特征的问题。引入信息增益比
>
> ​	$I_R(D,A)=\frac{I(A,D)}{H_A(D)}$
>
> ​	其中D为样本特征输出的集合，A为样本特征，则特征熵为：
>
> ​	$H_A(D)=-\sum_i^n\frac{|D_i|}{|D|}log_2\frac{|D_i|}{|D|}$
>
> ​	$其中n为特征A的类别数， |Di|为特征A取第i个值时对应的样本个数。|D|为总样本个数$
>
> 对于**第三个缺失值处理的问题**，主要需要解决的是两个问题，一是在样本某些特征缺失的情况下选择划分的属性，二是选定了划分属性，对于在该属性上缺失特征的样本的处理。
>
> ​	对于第一个子问题，对于某一个有缺失特征值的特征A。C4.5的思路是将数据分成两部		 	分，对每个样本设置一个权重（初始可以都为1），然后划分数据，一部分是有特征值A的数据D1，另一部分是没有特征A的数据D2. 然后对于没有缺失特征A的数据集D1来和对应的A特征的各个特征值一起计算加权重后的信息增益比，最后乘上一个系数，这个系数是无特征A缺失的样本加权后所占加权总样本的比例。
>
> ​	对于第二个子问题，可以将缺失特征的样本同时划分入所有的子节点，不过将该样本的权重按各个子节点样本的数量比例来分配。比如缺失特征A的样本a之前权重为1，特征A有3个特征值A1,A2,A3。 3个特征值对应的无缺失A特征的样本个数为2,3,4.则a同时划分入A1，A2，A3。对应权重调节为2/9,3/9, 4/9。
>
> 
>
> 对于**第四个问题**，C4.5引入了正则化系数进行初步的剪枝
>
> 
>
> **不足**
>
> 1)由于决策树算法非常容易过拟合，因此对于生成的决策树必须要进行剪枝。剪枝的算法有非常多，C4.5的剪枝方法有优化的空间。思路主要是两种，一种是预剪枝，即在生成决策树的时候就决定是否剪枝。另一个是后剪枝，即先生成决策树，再通过交叉验证来剪枝。后面在下篇讲CART树的时候我们会专门讲决策树的减枝思路，主要采用的是后剪枝加上交叉验证选择最合适的决策树。
>
> 2)C4.5生成的是多叉树，即一个父节点可以有多个节点。很多时候，在计算机中二叉树模型会比多叉树运算效率高。如果采用二叉树，可以提高效率。
>
> 3)C4.5只能用于分类，如果能将决策树用于回归的话可以扩大它的使用范围。
>
> 4)C4.5由于使用了熵模型，里面有大量的耗时的对数运算,如果是连续值还有大量的排序运算。如果能够加以模型简化可以减少运算强度但又不牺牲太多准确性的话，那就更好了。



##### CART算法

​	CART分类树算法使用基尼系数来代替信息增益比，基尼系数代表了模型的不纯度，基尼系数越小，则不纯度越低，特征越好

​	$Gini(D)=\sum\limits_{k=1}^Kp_k(1-p_k)=1-\sum p_k^2$

​	$其中p_k为任一样本点属于第k类的概率，也可以说成样本数据集中属于k类的样本的比例$

在特征A条件下的集合D的基尼指数为 :

​	$Gini(D|A)=\sum\frac{|D_i|}{|D|}Gini(D_i)$



| 算法 | 支持模型   | 树结构 | 特征选择         | 连续值处理 | 缺失值处理 | 剪枝   |
| ---- | ---------- | ------ | ---------------- | ---------- | ---------- | ------ |
| ID3  | 分类       | 多叉树 | 信息增益         | 不支持     | 不支持     | 不支持 |
| C4.5 | 分类       | 多叉树 | 信息增益比       | 支持       | 支持       | 支持   |
| CART | 分类、回归 | 二叉树 | 基尼系数、均方差 | 支持       | 支持       | 支持   |





## 朴素贝叶斯分类



##### 朴素

​	指各个事件独立







## 支持向量机



##### 基本思想

​	首先通过非线性变换将输入空间变换到一个高维的空间，然后在这个新的空间求最优分类面即最大间隔分类面，而这种非线性变换是通过定义适当的内积核函数来实现的。SVM实际上是根据统计学习理论依照结构风险最小化的原则提出的，要求实现两个目的：1）两类问题能够分开（经验风险最小）2）margin最大化（风险上界最小）既是在保证风险最小的子集中选择经验风险最小的函数。



##### 超参数

 - C参数。$Error = C * Classification Error + Margin Error​$.  C越大，则重视正确的分类；C越小，越重视合适的间隔。
 - 内核$(kernel)​$. 多项式内核(mutinomial  kernel)（degree参数，最大单项式次数，其越大，会过拟合，反之会欠拟合)）, RBF kernel($\gamma参数​$，其越大，会过拟合，反之会欠拟合)





## 集成方法（AdaBoost）



##### 基本算法思路

​	1.随机选取数据样本，以高准确率为目标进行分类，获得一个学习结果，计算学习结果的权重值；

​	2.以上一步骤获得的学习结果为参考，对错误点进行惩罚记录,使得错误点权重和正确点权重相同；

​	3.重新随机选取数据，重复步骤1，直到达到限制条件满足；

​	4.根据每个学习结果的权重值，进行同区域的合并运算（权重值相加）；

​	5.步骤4会得到分类区域，根据样本所在区域进行预测



##### 超参数

 - base_estimator: 弱学习器使用的模型
 - n_estimators:使用弱学习器的最大数量



## K-MEAN算法



##### 应用场景

 - 实现同兴趣的用户分类，预测其可能感兴趣的事物



##### 局限

 - 依赖聚类中心初始点的位置，容易得出局部最小值





## 层次聚类（sklearn：AgglomerativeClustering）



##### 单连接聚类法

​	两类的最近点之间的距离来判别是否同类

​	问题：容易囊括大部分点

##### 全连接聚类法

​	两类的最远点之间的距离来判别是否同类

​	问题：容易囊括大部分点



##### 超参数

 - n_clusters
 - linkage: 连接方法，即计算距离的方法，$ward$指的是离差平方和方法（预设）, $complete$指的是全连接，$average$指的是平均值



##### 层次树（系统树）

```python
from scipy.cluster.hierarchy import dendrogram, ward, single
from sklearn import datasets
import matplotlib.pyplot as plt
# Load dataset
X = datasets.load_iris().data[:10]
# Preform clustering
linkage_matrix = ward(X)
# Plot dendogram
dendogram(likage_matrix)

plt.show()
```



##### 优点

- 聚类结构可视化
- 结果层次表达，信息丰富
- 对于数据内部有层次结构的，结果更好

缺点

 - 对于噪声点或者是离群值很敏感
 - 计算量大，$O(N^2)$





## 密度聚类（DBSCAN）



##### 超参数

 - eps：距离
 - min_sample：最小样本
 - 



##### 优点

- 在处理有噪声的数据集方面作用巨大
- 不需要预设类的个数
- 不局限于某种外形的类，可以灵活的找到并分离各种形状和大小的类



##### 缺点

- 边界点有可能在每次结果属于不同的类
- 在找到不同密度的类方面有一定难度



## 高斯混合模型（GNN）聚类



##### 假设

​	样本分布符合高斯分布



##### 算法流程

  - 初始化K个高斯分布模型
  - 将数据软聚类成我们初始化的两个高斯（期望步骤或E步骤），即计算对每个模型的隶属度
  - 基于软聚类重新估计高斯（最大化步骤或M步骤）
  - 评估对数似然来检查收敛，收敛则返回结果，否则回到第二步，反复进行



##### sklearn使用简单示例

```python
from sklearn import datasets, mixture

X = datasets.load_iris().data[:10]

gmm = mixture.GaussianMixture(n_components=3)
gmm.fit(X)

clustering = gmm.predict(X)
```



##### 优点

 - 提供软聚类（软聚类是多个聚类的示例性隶属度）
 - 聚类外观灵活



##### 缺点

- 对初始化值敏感
- 有可能收敛到局部最优
- 收敛速度慢



##### 应用场景

- 信号分离（音频、加速度，速度，分类天文学的脉冲星）
- 生物识别（签名等）
- 图像识别（流式视频背景还原、人员追踪）

​	Paper: [Nonparametric discovery of human routines from sensor data](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.681.3152&rep=rep1&type=pdf) [PDf]

​	Paper: [Application of the Gaussian mixture model in pulsar astronomy](https://arxiv.org/abs/1205.6221) [PDF]

​	Paper: [Speaker Verification Using Adapted Gaussian Mixture Models](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.117.338&rep=rep1&type=pdf) [PDF]

​	Paper: [Adaptive background mixture models for real-time tracking](http://www.ai.mit.edu/projects/vsam/Publications/stauffer_cvpr98_track.pdf) [PDF]

​	Video: <https://www.youtube.com/watch?v=lLt9H6RFO6A>





## 主成分分析（PCA)



##### 主成分确定方式

 - 找到最大方差方向
 - 原因：
    - 可以最大程度保留原始数据的信息量



### 使用场景

- 想要访问隐藏的特征，认为这些隐藏的特征显示在你的数据的图案中
- 降维
  - 可视化高维数据（用前两维）
  - 减少噪音
  - 让算法计算或工作更好更快



##### sklearn中使用

```python
from sklearn.decomposition import PCA
pca = PCA(n_componects=2)
pca.fit(X)

pca.explained_variance_ratio_
first_pc = pca.components_[0]
second_pc = pca.components_[1]
```





## 随机投影



##### 优点

 - 在计算上比PCA更有效率
 - 在数据太多维度，PCA无法直接接计算的情景下，可以使用其



##### 相关参数

 - eps：可接受的投影误差

   $(1-eps)||u-v||^2 <||p(u)-p(v)||^2<(1+eps)||u-v||^2$

   随机投影后的维度使用用$eps$来计算的，可以接受的$eps$越大，压缩率越大



##### sklearn使用方法

```python
from sklearn.random_projection import SparseRandomProjection()
# 可以调整eps或者输入目标组件
rp = SparseRandomProjection()
# 将数据集压缩
new_X = rp.fit_transform(X)
```







## 独立成分分析（ICA）



##### 假设

- 各成分是分别统计的
- 成分必须为非高斯分布



###### 使用场景

 - 分离混合的声音（盲源分离问题）
 - 医学扫描仪
 - 金融数据分析（论文层面的）



##### sklearn使用方法

```python
from sklearn.decomposition import FastICA

# Each 'signal' variable is an array, e.g. audio waveform
X = list(zip(signal_1, signal_2, signal_3))

ica = FastICA(n_components=3)

components = ica.fit_transform(X)
```







## 神经网络

 ##### 交叉熵

- 两类别交叉熵

  $Cross-Entropy=-\sum\limits_{i=1}^{m}y_iln(p_i) + (1-y_i)ln(1-p_i)$

   -- $y_i$ = 1 if true else 0

   -- $p_i$是第$i$个样本

  

- 多类别交叉熵

  $Cross-Entropy=-\sum\limits_{i=1}^{n}\sum\limits_{j=1}^{m}y_{ij}ln(p_{ij})$

   -- $i特征类别，j样本点​$

   -- $p_{ij}为第j个样本是i特征的概率​$

   -- $y_{ij}$ = 1 if true else 0

##### 梯度计算

在上几个视频中，我们了解到为了最小化误差函数，我们需要获得一些导数。我们开始计算误差函数的导数吧。首先要注意的是 s 型函数具有很完美的导数。即

\sigma'(x) = \sigma(x) (1-\sigma(x))*σ*′(*x*)=*σ*(*x*)(1−*σ*(*x*))

原因是，我们可以使用商式计算它：



![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/ba81c06c-40be-4ae9-b557-cc0f74cd4116)



现在，如果有 m*m* 个样本点，标为 $x^{(1)}, x^{(2)}, \ldots, x^{(m)}​$, 误差公式是：

$E = -\frac{1}{m} \sum_{i=1}^m \left( y^{(i)} \ln(\hat{y^{(i)}}) + (1-y^{(i)}) \ln (1-\hat{y^{(i)}}) \right)$

预测是$ \hat{y^{(i)}} = \sigma(Wx^{(i)} + b).$

我们的目标是计算 E,*, 在单个样本点 x 时的梯度（偏导数），其中 x 包含 n 个特征，即$x = (x_1, \ldots, x_n).$

$\nabla E =\left(\frac{\partial}{\partial w_1}E, \cdots, \frac{\partial}{\partial w_n}E, \frac{\partial}{\partial b}E \right)​$

为此，首先我们要计算 $\frac{\partial}{\partial w_j} \hat{y}​$.

$\hat{y} = \sigma(Wx+b)​$, 因此：



![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/cfe9e171-2608-4c05-a1bb-f9a7d1a5eee1)



最后一个等式是因为和中的唯一非常量项相对于 $w_j*x_j,明显具有导数​$

现在可以计算 $\frac {\partial} {\partial w_j}E $

![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/ccfebc74-13ff-48a8-9d8c-3562f5b9945b)



类似的计算将得出：（备注：下图公式缺少一个负号，且其为 m 个样本点时的公式）

【针对单个样本点时，E 对 b 求偏导的公式为：\frac {\partial} {\partial b} E=-(y -\hat{y})∂*b*∂*E*=−(*y*−*y*^)】



![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/936e53ac-6b05-436e-bbc9-9f5a01e82a0a)



这个实际上告诉了我们很重要的规则。对于具有坐标$ (x_1, \ldots, x_n)$的点，标签 *y*, 预测$ \hat{y}$该点的误差函数梯度是$ \left(-(y - \hat{y})x_1, \cdots, -(y - \hat{y})x_n, -(y - \hat{y}) \right)$

总之

$\nabla E(W,b) = -(y - \hat{y}) (x_1, \ldots, x_n, 1)$

如果思考下，会发现很神奇。梯度实际上是标量乘以点的坐标！什么是标量？也就是标签和预测直接的差别。这意味着，如果标签与预测接近（表示点分类正确），该梯度将很小，如果标签与预测差别很大（表示点分类错误），那么此梯度将很大。请记下：小的梯度表示我们将稍微修改下坐标，大的梯度表示我们将大幅度修改坐标。

如果觉得这听起来像感知器算法，其实并非偶然性！稍后我们将详细了解。





##### 和感知器算法区别

- 对错误点的处理不同
  - 感知器算法对错误分类点，会要求超平面靠近；正确分类的点不做处理
  - 神经网络则都会更新权重，但分类正确的权重变化小，分类错误的权重变化多
- 更新权重方法不同
  - 神经网络，$w_i$ to $w_i + \alpha(y-\hat{y})x_i$
  - 感知器算法，$w_i$ to $w_i ± \alpha x_i$
  - $\hat{y}取值也不同$



## 深度神经网络（MLP）

##### 前向反馈

 - 前向反馈是神经网络用来将输入变成输出的流程
 - 误差函数
    - ![](C:\Users\Barnett\Desktop\Udacity笔记\img\深度升级网络-前向传播-误差函数.png)

##### 反向传播

- 进行前向反馈运算。
- 将模型的输出与期望的输出进行比较。
- 计算误差。
- 向后运行前向反馈运算（反向传播），将误差分散到每个权重上。
- 更新权重，并获得更好的模型。
- 继续此流程，直到获得很好的模型。



##### Keras构建

```python
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

#create the Sequential model
model = Sequential()
```

[keras.models.Sequential](https://keras.io/models/sequential/) 类是神经网络模型的封装容器。它会提供常见的函数，例如 `fit()`、`evaluate()`和 `compile()`。

```python
# Continuously above

# 第一层 - 添加有128个节点的全连接层以及32个节点的输入层
model.add(Dense(128, input_dim=32))
# 第二层 - 添加softmax 激活层
model.add(Activation('softmax'))
# 第三层 - 添加全连接层
model.add(Dense(10))
# 第四层 - 添加Sigmoid激活层
modeladd(Activation('sigmoid'))
```

构建好模型后，我们就可以用以下命令对其进行编译。我们将损失函数指定为我们一直处理的 `categorical_crossentropy`

```python
model.compile(loss='categorical_crossentory', optimzer='adam', metric=['accuracy'])
```

查看模型的架构

```python
model.summary()
```

然后使用以下命令对其进行拟合，指定 epoch 次数和我们希望在屏幕上显示的信息详细程度。

然后使用fit命令训练模型并通过 epoch 参数来指定训练轮数（周期），每 epoch 完成对整数据集的一次遍历。 verbose 参数可以指定显示训练过程信息类型，这里定义为 0 表示不显示信息。

```python
model.fit(X, y, epochs=1000, verbose)
```

最后，我们可以使用以下命令来评估模型：

```python
score = model.evaluate(X, y)
```

*补*：![](.\img\softmax和sigmoid的异同.png)



##### 训练优化

 - 见模型复杂度图表



##### 正则化

 ![](.\img\深度神经网络 - 正则化.png)



##### dropout机制

 - 每个epoch随机选取某个百分的数量的节点不参与计算



##### 避免到局部最低点

 - 改变激活函数

   $tanh(x)=\frac{e_x-e_{-x}}{e_x+e_{-x}}​$

   $$relu=\begin{cases}x, & x >=0\\0 , & x<0 \end{cases}$$

- 随机重新开始（随机初始点多次来对比）

- 用动量和决心快速移动

  $\beta 动量，在(0, 1]中$

  $STEP(n) -> STEP(n) +\beta STEP(n-1) +\beta STEP(n-2) + ...  ​$

  ![](.\img\深度学习 - 动量.png)

##### 

##### 何时效果不错

 	相比于CNN，在处理的图片是简单的图形，没有更深层次的图形相对位置时，可以得到相对较好的结果（MLP会将图片转换成一维向量处理），反之，当识别的图形依赖于更深层的图像识别时，MLP识别的结果会比较差





## 卷积神经网络（CNN）

##### 相对于MLP的改进

| MLPs                                                  | CNNs                                                         |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| 仅用全连接层连接（导致28*28的图片都会有相当多的参数） | 通过使用更加稀疏互联的层级来解决左侧问题，层级之间的连接，由图像矩阵的二维结构决定 |
| 仅接受向量，丢失了二维（或更高维度）的空间信息        | 也接受矩阵输入，可以保留空间信息                             |



##### 卷积层工作示意

![](https://s3.cn-north-1.amazonaws.com.cn/u-img/66fd64c5-754f-4ce7-b47f-185687265cc6)



​					                        *窗口为3x3，stride为1的卷积层*

![](https://s3.cn-north-1.amazonaws.com.cn/u-img/e2b648e7-06f3-4ef9-bea4-7e2758547f47)

​							*窗口为3x3，stride为1的卷积层*

##### Keras操作示意

```python
from keras.layers import Conv2D
Conv2D(filters, kernel_size, strides, padding, activation='relu', input_shape)
```

> 必须传递以下参数：
>
> - `filters` - 过滤器数量。
> - `kernel_size` - 指定（方形）卷积窗口的高和宽的数字。
>
> 你可能还需要调整其他可选参数：
>
> - `strides` - 卷积 stride。如果不指定任何值，则 `strides` 设为 `1`。
> - `padding` - 选项包括 `'valid'` 和 `'same'`。如果不指定任何值，则 `padding` 设为 `'valid'`。
> - `activation` - 通常为 `'relu'`。如果未指定任何值，则不应用任何激活函数。**强烈建议**你向网络中的每个卷积层添加一个 ReLU 激活函数。
>
> **注意**：可以将 `kernel_size` 和 `strides` 表示为数字或元组。
>
> 在模型中将卷积层当做第一层级（出现在输入层之后）时，必须提供另一个 `input_shape` 参数：
>
> - `input_shape` - 指定输入的高度、宽度和深度（按此顺序）的元组。
>
> **注意**：如果卷积层*不是*网络的第一个层级，***请勿***包含 `input_shape` 参数。
>
> 你还可以设置很多其他元组参数，以便更改卷积层的行为。要详细了解这些参数，建议参阅官方[文档](https://keras.io/layers/convolutional/)。

**e.g.1**

> 假设我要构建一个 CNN，输入层接受的是 200 x 200 像素（对应于高 200、宽 200、深 1 的三维数组）的灰度图片。然后，假设我希望下一层级是卷积层，具有 16 个过滤器，每个宽和高分别为 2。在进行卷积操作时，我希望过滤器每次跳转 2 个像素。并且，我不希望过滤器超出图片界限之外；也就是说，我不想用 0 填充图片。要构建该卷积层，我将使用下面的代码：
>
> ```python
> Conv2D(filters=16, kernel_size=2, strides=2, activation='relu', input_shape=(200, 200, 1))
> ```

**e.g.2**

> 假设我希望 CNN 的下一层级是卷积层，并将示例 1 中构建的层级作为输入。假设新层级是 32 个过滤器，每个的宽和高都是 3。在进行卷积操作时，我希望过滤器每次移动 1 个像素。我希望卷积层查看上一层级的所有区域，因此不介意过滤器在进行卷积操作时是否超过上一层级的边缘。然后，要构建此层级，我将使用以下代码：
>
> ```python
> Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
> ```

**e.g.3**

> 以下代码表示：有 64 个过滤器，每个的大小是 2x2，层级具有 ReLU 激活函数。层级中的其他参数使用默认值，因此卷积的 stride 为 1，填充设为 'valid'。
>
> ```python
> COnv2D(64, (2, 2), activation='relu')
> ```



##### 卷积层中的参数数量

卷积层中的参数数量取决于 `filters`、`kernel_size` 和 `input_shape` 的值。我们定义几个变量：

- `K` - 卷积层中的过滤器数量
- `F` - 卷积过滤器的高度和宽度
- `D_in` - 上一层级的深度

注意：`K` = `filters`，`F` = `kernel_size`。类似地，`D_in` 是 `input_shape` 元组中的最后一个值。

因为每个过滤器有 `F*F*D_in` 个权重，卷积层由 `K` 个过滤器组成，因此卷积层中的权重总数是 `K*F*F*D_in`。因为每个过滤器有 1 个偏差项，卷积层有 `K` 个偏差。因此，卷积层中的**参数数量**是 `K*F*F*D_in + K`。



##### 卷积层的形状

卷积层的形状取决于 `kernel_size`、`input_shape`、`padding` 和 `stride` 的值。我们定义几个变量：

- `K` - 卷积层中的过滤器数量
- `F` - 卷积过滤器的高度和宽度
- `H_in` - 上一层级的高度
- `W_in` - 上一层级的宽度

注意：`K` = `filters`、`F` = `kernel_size`，以及`S` = `stride`。类似地，`H_in` 和 `W_in` 分别是 `input_shape` 元组的第一个和第二个值。

卷积层的**深度**始终为过滤器数量 `K`。

如果 `padding = 'same'`，那么卷积层的空间维度如下：

- **height** = ceil(float(`H_in`) / float(`S`))
- **width** = ceil(float(`W_in`) / float(`S`))

如果 `padding = 'valid'`，那么卷积层的空间维度如下:

- **height** = ceil(float(`H_in` - `F` + 1) / float(`S`))
- **width** = ceil(float(`W_in` - `F` + 1) / float(`S`))

##### 

##### 池化层

将卷积层当做输入，压缩卷积层

- 最大池化层

  ![](https://s3.cn-north-1.amazonaws.com.cn/u-img/23b82900-48fd-4b99-9944-1ee5b28ec700)

  ```python
  from keras.layers import MaxPooling2D
  
  MaxPooling2D(pool_size, strides, padding)
  ```

  ###### 参数

  你必须包含以下参数：

  - `pool_size` - 指定池化窗口高度和宽度的数字。

  你可能还需要调整其他可选参数：

  - `strides` - 垂直和水平 stride。如果不指定任何值，则 `strides` 默认为 `pool_size`。
  - `padding` - 选项包括 `'valid'` 和 `'same'`。如果不指定任何值，则 `padding` 设为 `'valid'`。

  **注意**：可以将 `pool_size` 和 `strides` 表示为数字或元组。

  此外，建议阅读官方[文档](https://keras.io/layers/pooling/#maxpooling2d)。



##### 图片分类的CNN层级排列方式

- 固定的图片大小（通常为正方形）

- 架构设计目标是获取该输入，然后逐渐使其深度大于宽和高（卷积层将增加深度，最大池化层将用于减少空间维度）

  **不含最大池化层架构及参数：**

  ![](.\img\图片分类CNN.png)

  ![](.\img\不含最大池化层参数.png)

  **加入最大池化层架构和参数变化**：

  ![](.\img\图片分类CNN_加入最大池化层架构.png)

- 使用最后一层最大池化层，来采用全连接层，来进行预测

  ![](.\img\CNN结果预测.png)

  ![](.\img\CNN结果预测代码.png)

  





##### Keras的图片增强功能

- 载入CIFAR-10数据集
- 可视化前24张图片
- 通过像素除以255，标准化像素数据
- 将数据集分成训练、验证和测试三部分

- 配置参数增强图片生成器

```python
from keras.preprocessing.image import ImageDataGenerator

# create and configure augmented image generator
datagen_train = ImageDataGenerator(
    width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
    height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
    horizontal_flip=True) # randomly flip images horizontally

# fit augmented image generator on data
datagen_train.fit(x_train)
```

- 定义模型架构

```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.summary()
```

- 编译模型

```python
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                  metrics=['accuracy'])
```

- 添加图片增强功能并训练

  - 注意使用`~.fig_generator`方法

  - 注意使用`~.flow`方法来增强图片

  - ###### 

```python
from keras.callbacks import ModelCheckpoint   

batch_size = 32
epochs = 100

# train the model
checkpointer = ModelCheckpoint(filepath='aug_model.weights.best.hdf5', verbose=1, 
                               save_best_only=True)
model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs, verbose=2, callbacks=[checkpointer],
                    validation_data=(x_valid, y_valid),
                    validation_steps=x_valid.shape[0] // batch_size)
```

> 关于 `steps_per_epoch` 的注意事项
>
> `fit_generator` 具有很多参数，包括
>
> ```python
> steps_per_epoch = x_train.shape[0] / batch_size
> ```
>
> 其中 `x_train.shape[0]` 对应的是训练数据集 `x_train` 中的独特样本数量。通过将 `steps_per_epoch` 设为此值，我们确保模型在每个 epoch 中看到 `x_train.shape[0]` 个增强图片。

- 载入最佳模型权重
- 计算在测试集上的准确度



##### 迁移学习

> 迁移学习是指对提前训练过的神经网络进行调整，以用于新的不同数据集。
>
> 取决于以下两个条件：
>
> - 新数据集的大小，以及
> - 新数据集与原始数据集的相似程度
>
> 使用迁移学习的方法将各不相同。有以下四大主要情形：
>
> 1. 新数据集很小，新数据与原始数据相似
> 2. 新数据集很小，新数据不同于原始训练数据
> 3. 新数据集很大，新数据与原始训练数据相似
> 4. 新数据集很大，新数据不同于原始训练数据
>
> 为了解释每个情形的工作原理，我们将以一个普通的预先训练过的卷积神经网络开始，并解释如何针对每种情形调整该网络。我们的示例网络包含三个卷积层和三个完全连接层：
>
> 
>
> ![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/2fcc5caf-46c3-4915-ac20-9696960fb9b7)
>
> 神经网络的一般概述
>
> 
>
> 下面是卷积神经网络的作用一般概述：
>
> - 第一层级将检测图片中的边缘
> - 第二层级将检测形状
> - 第三个卷积层将检测更高级的特征
>
> 每个迁移学习情形将以不同的方式使用预先训练过的神经网络。
>
> 
>
> ### 情形 1：小数据集，相似数据
>
> 
>
> ![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/85c5bf9f-6e61-42c2-b084-f73360fc128b)
>
> 情形 1：具有相似数据的小数据集
>
> 
>
> 如果新数据集很小，并且与原始训练数据相似：
>
> - 删除神经网络的最后层级
> - 添加一个新的完全连接层，与新数据集中的类别数量相匹配
> - 随机化设置新的完全连接层的权重；冻结预先训练过的网络中的所有权重
> - 训练该网络以更新新连接层的权重
>
> 为了避免小数据集出现过拟合现象，原始网络的权重将保持不变，而不是重新训练这些权重。
>
> 因为数据集比较相似，每个数据集的图片将具有相似的更高级别特征。因此，大部分或所有预先训练过的神经网络层级已经包含关于新数据集的相关信息，应该保持不变。
>
> 以下是如何可视化此方法的方式：
>
> 
>
> ![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/47d819b4-2472-4969-a068-4125b946a937)
>
> 具有小型数据集和相似数据的神经网络
>
> 
>
> ### 情形 2：小型数据集、不同的数据
>
> 
>
> ![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/3d0fbbd6-c73d-496c-be1c-a63e2f655122)
>
> 情形 2：小型数据集、不同的数据
>
> 
>
> 如果新数据集很小，并且与原始训练数据不同：
>
> - 将靠近网络开头的大部分预先训练过的层级删掉
> - 向剩下的预先训练过的层级添加新的完全连接层，并与新数据集的类别数量相匹配
> - 随机化设置新的完全连接层的权重；冻结预先训练过的网络中的所有权重
> - 训练该网络以更新新连接层的权重
>
> 因为数据集很小，因此依然需要注意过拟合问题。要解决过拟合问题，原始神经网络的权重应该保持不变，就像第一种情况那样。
>
> 但是原始训练集和新的数据集并不具有相同的更高级特征。在这种情况下，新的网络仅使用包含更低级特征的层级。
>
> 以下是如何可视化此方法的方式：
>
> 
>
> ![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/298a5f0f-ac69-4581-b009-1a5763bef338)
>
> 具有小型数据集、不同数据的神经网络
>
> 
>
> ### 情形 3：大型数据集、相似数据
>
> 
>
> ![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/a0ea6989-608b-43e1-a571-b25d633513d7)
>
> 情形 3：大型数据集、相似数据
>
> 
>
> 如果新数据集比较大型，并且与原始训练数据相似：
>
> - 删掉最后的完全连接层，并替换成与新数据集中的类别数量相匹配的层级
> - 随机地初始化新的完全连接层的权重
> - 使用预先训练过的权重初始化剩下的权重
> - 重新训练整个神经网络
>
> 训练大型数据集时，过拟合问题不严重；因此，你可以重新训练所有权重。
>
> 因为原始训练集和新的数据集具有相同的更高级特征，因此使用整个神经网络。
>
> 以下是如何可视化此方法的方式：
>
> 
>
> ![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/5813bdee-1d46-4188-88c2-971967496348)
>
> 具有大型数据集、相似数据的神经网络
>
> 
>
> ### 情形 4：大型数据集、不同的数据
>
> 
>
> ![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/09d40fc9-6815-4ce3-9469-cddb99ce07b0)
>
> 情形 4：大型数据集、不同的数据
>
> 
>
> 如果新数据集很大型，并且与原始训练数据不同：
>
> - 删掉最后的完全连接层，并替换成与新数据集中的类别数量相匹配的层级
> - 使用随机初始化的权重重新训练网络
> - 或者，你可以采用和“大型相似数据”情形的同一策略
>
> 虽然数据集与训练数据不同，但是利用预先训练过的网络中的权重进行初始化可能使训练速度更快。因此这种情形与大型相似数据集这一情形完全相同。
>
> 如果使用预先训练过的网络作为起点不能生成成功的模型，另一种选择是随机地初始化卷积神经网络权重，并从头训练网络。
>
> 以下是如何可视化此方法的方式：
>
> 
>
> ![img](https://s3.cn-north-1.amazonaws.com.cn/u-img/061f19fd-25b0-40bf-bacd-4451edcbb20b)
>
> 具有大型数据集、不同数据的网络
>



## 强化学习



### 强化学习框架

##### 问题

- 设置

![](.\img\强化学习_设置.png)

  		1. $S_0$表示在时间步0状态的状态
  		2. $A_0$表示在时间步0智能体做出的动作
  		3. 在$A_0$动作之后产生的在时间步1的状态$S_1$
  		4. 在$A_0$动作产生的时间步1的奖励$R_1$
  		5. 以此类推$S_0, A_0, R_1, S_1, A_1, R_2, S_2, A_2, ... ,R_t, S_t, A_t$



- 阶段性任务与连续性任务

  - **任务**是一种强化学习问题。
  - **连续性任务**是一直持续下去、没有结束点的任务。
  - **阶段性任务**是起始点和结束点明确的任务。
  - 在这种情况下，我们将一个完整的互动系列（从开始到结束）称为一个**阶段**。
  - 每当智能体抵达**最终状态**，阶段性任务都会结束。

- 奖励假设

  - 智能体的目标始终可以描述为最大化期望累积奖励，称之为奖励假设

- 折扣回报

  $G_t = R_{t+1} + \gamma R_{t+2}+ \gamma^2 R_{t+3}+ \gamma^3 R_{t+4}+...$

  -- $\gamma$为折扣率，[0, 1]

  -- $R$为奖励

  -- $G$折扣汇报

- 一步动态特性

  在随机时间步 *t*，智能体环境互动变成一系列的状态、动作和奖励。

  $(S_0, A_0, R_1, S_1, A_1, \ldots, R_{t-1}, S_{t-1}, A_{t-1}, R_t, S_t, A_t)​$

  当环境在时间步 $t+1$ 对智能体做出响应时，它只考虑上一个时间步$ (S_t, A_t)$ 的状态和动作。

  尤其是，它不关心再上一个时间步呈现给智能体的状态。（*换句话说*，环境不考虑任何 $\{S_0, \ldots, S_{t-1}\}​$。）

  并且，它不考虑智能体在上个时间步之前采取的动作。（*换句话说*，环境不考虑任何 $\{ A_0, \ldots, A_{t-1}\}​$。）

  此外，智能体的表现如何，或收集了多少奖励，对环境选择如何对智能体做出响应没有影响。（*换句话说*，环境不考虑任何$\{R_0, \ldots, R_t\}$。）

  因此，我们可以通过指定以下设置完全定义环境如何决定状态和奖励

  $p(s',r|s,a) \doteq \mathbb{P}(S_{t+1}=s', R_{t+1}=r|S_t = s, A_t=a)​$

  对于每个可能的 s', r, s, \text{and } a*s*′,*r*,*s*,and *a*。这些条件概率用于指定环境的**一步动态特性**。

- MDP(Markov Decision Process马尔科夫决策法) Definition

  ![](.\img\强化学习_MDP Definition.png)

- 总结

  ![强化学习_智能体和环境互动](.\img\强化学习_智能体和环境互动.png)



##### 解决问题

- 贝尔曼预期方程含义

![](.\img\强化学习_Bellman Expectation Equation.png)

- 贝尔曼预期方程和MDP的差异

![](.\img\强化学习_MDP和贝尔曼预期方程的差异.png)

- 不同策略的计算方法

![](.\img\强化学习_不同策略的计算方法.png)

- 总结

![](C:\Users\Barnett\Desktop\Udacity笔记\img\强化学习_高尔夫智能体的状态值函数.png)
![](.\img\强化学习_解决方案_0.png)

![](.\img\强化学习_解决方案_1.png)