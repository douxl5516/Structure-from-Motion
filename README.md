# Camera Position Reckoning based on SfM

## 目标
实现基于SfM算法的，以目标拍摄效果为导向的相机位置反演系统。

## 实现方法
- 读取视频，取间隔帧数的照片作为SfM算法的关键帧。
- 取视频的最后一张图片，在GUI上选框进行投影变换，实现目标图像的获取。
- 将所得到的序列图片运行SfM算法，进行三维重建。
- 将目标图片加入SfM算法序列，运行结束后能够获得相机的外参矩阵。

## 目前实现
- 利用[张正友标定法](https://www.computer.org/csdl/trans/tp/2000/11/i1330-abs.html)，采用[ting2696/zhang-s-method](ting2696/zhang-s-method)的棋盘格数据集实现相机标定。

## 待实现想法
- 根据[清晰度评价方法](https://blog.csdn.net/dcrmg/article/details/53543341)，提供目标函数，从输入视频中选取最清晰的图片。