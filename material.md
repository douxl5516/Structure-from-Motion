# 参考资料
## 算法原理及数学知识
- [Sfm方法过程及原理](https://blog.csdn.net/qq_33826977/article/details/79834735)
- [本质矩阵](https://blog.csdn.net/j10527/article/details/51295099)
>本质方程与基础方程都是刻画双目视觉极几何关系的两个方程，只不过它们所使用的坐标系不一样而已，本质方程是在相机光心坐标系下描述的双目成像几何关系，而基础方程是本质方程在相机像素坐标系下的表示。
本质矩阵是平移量t叉乘旋转矩阵R的结果。所以通过本质方程估计出本质矩阵，然后再对本质矩阵进行运动分解，即可得到双目摄像头之间的相互位置关系，即双目的标定。值得一提的是，由于本质矩阵的秩为5，分解出来的平移向量t只能得到方向，其模需要使用其它方法来获取。

## SfM的实现
### OpenCV实现SfM(步骤、知识点及部分源码)
- [OpenCV实现SfM（一）：相机模型](https://blog.csdn.net/AIchipmunk/article/details/48132109)
- [OpenCV实现SfM（二）：双目三维重建](https://blog.csdn.net/AIchipmunk/article/details/48157369)
- [OpenCV实现SfM（三）：多目三维重建](https://blog.csdn.net/aichipmunk/article/details/51232861)
- [OpenCV实现SfM（四）：Bundle Adjustment](https://blog.csdn.net/aichipmunk/article/details/52433884)

### 求解经验、方法及知识总结
- [SfM总结二](https://blog.csdn.net/aaron121211/article/details/52265851)

### SfM实现效果及展示
- [Structure from motion（SFM）原理 - 附我的实现结果(无源码)](https://blog.csdn.net/Mahabharata_/article/details/70799695)
- [【计算机视觉】从运动中恢复结构SfM-场景重建,三维重建(无源码)](https://blog.csdn.net/KYJL888/article/details/72843001)
- [单目三维重建(有源码)](https://blog.csdn.net/sinat_39411798/article/details/80453642)

## 已实现的三维重建项目
- [SfM-Toy-Library](https://github.com/royshil/SfM-Toy-Library)

## 关于投影变换
- [CS全能扫描王](https://www.cnblogs.com/skyfsm/p/7324346.html)
- [OpenCV Java 实现票据、纸张的四边形边缘检测与提取、摆正](https://www.cnblogs.com/josephkim/p/8319069.html)

## 数据集
- [数据集](https://github.com/awesomedata/awesome-public-datasets)
- [树洞图片数据集](https://github.com/alicevision/dataset_monstree)
- [佛像图片数据集](https://github.com/alicevision/dataset_buddha)
- [SfM_quality_evaluation（含相机内参和重建结果）](https://github.com/alicevision/SfM_quality_evaluation)
- [Middlebury数据集（含相机内参和重建结果）](http://vision.middlebury.edu/mview/data/)
- [ComputerVisionDatasets(最全最多)](https://github.com/AIBluefisher/ComputerVisionDatasets)

## 三维重建点云效果的查看 
- [C++ PCL库:PCLVisualizer](http://docs.pointclouds.org/trunk/classpcl_1_1visualization_1_1_p_c_l_visualizer.html)
- [OpenCV自带viz](https://docs.opencv.org/master/de/dfd/tutorial_sfm_import_reconstruction.html)
