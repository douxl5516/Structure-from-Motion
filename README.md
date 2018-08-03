# Camera Position Reckoning based on SfM

[TOC]

## 目标
实现基于SfM算法的，以目标拍摄效果为导向的相机位置反演系统。

## 基于环境
- [JDK8 x64](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)
- [openCV 3.4.2](https://opencv.org/releases.html)

## 实现方法
- 相机标定采用[GML C++ Camera Calibration Toolbox](http://graphics.cs.msu.ru/en/node/909)实现。
- 读取视频，取间隔帧数的照片作为SfM算法的关键帧。
- 取视频的最后一张图片，在GUI上选框进行投影变换，实现目标图像的获取。
- 将所得到的序列图片运行SfM算法，进行三维重建。
- 使用MATLAB查看三维重建点云效果。
```matlab
    P=[
        7.9932642, -6.161675, 18.40271;
        4.970612, -4.9678035, 11.996756;
        -0.35880902, 0.89857519, 7.5484924;
        0.78789729, 3.1232572, 6.3487396;...
    ];
    
    X=P(:,1);Y=P(:,2);Z=P(:,3);

    scatter3(X,Y,Z,'B.')
```
- 将目标图片加入SfM算法序列，运行结束后能够获得相机的外参矩阵。

## 目前实现
- 利用[张正友标定法](https://www.computer.org/csdl/trans/tp/2000/11/i1330-abs.html)，采用[ting2696/zhang-s-method](ting2696/zhang-s-method)的棋盘格数据集实现相机标定。（弃用）
- 使用SIFT算法实现了特征点的提取和匹配
- 实现了两张图像的三维重建，但精度不高。

## 待实现想法
- 根据[清晰度评价方法](https://blog.csdn.net/dcrmg/article/details/53543341)，提供目标函数，从输入视频中选取最清晰的图片。