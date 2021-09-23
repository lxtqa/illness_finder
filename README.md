# README

- 本项目借鉴自CSDN项目基于卷积神经网络的图像识别 之 火焰识别  

​		https://blog.csdn.net/cumtwys/article/details/104822560

## 运行方法

​		先运行illnessfinder.py，在sectionimages目录下生成section-inceptionV3-more21.h5文件，再运行illnesschecker.py，判断test目录下的样本情况，在终端输出。

## 运行环境

​		keras，tensorflow

## 建议

- 在本文件根目录下有已经运行好的section-inceptionV3-more21.h5文件，如果自己训练效果不佳，可以使用该文件来进行测试。

- 在本文件根目录下有作者自己电脑上运行的结果，可以作为参考。
- 在illnesschecker.py中可设定阈值，若训练不理想（指整体偏低或偏高），可以调整该值达到预期的效果。
- 由于样本数量过少，这使得程序运行结果不稳定且不准确，如果能够增加样本的数目，结果准确度会进一步提升。

- 将train和val中的文件进一步混合，效果会更加出色。
- 本文件和在作者电脑上运行时的train，val中的图片数目不同，这是为了更加贴近原作者的意图，作者猜测调整后可能会获得更好的结果。//_sectionimages中的文件为作者运行时的状态