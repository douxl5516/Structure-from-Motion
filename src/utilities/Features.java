package utilities;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.features2d.*;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import type.ImageData;

public class Features {
	/**
	 * 特征点提取函数
	 * 
	 * @param imgList     输入Mat的列表，对其中每一张图片进行特征点提取
	 * @param imgDataList 输出提取数据的列表，包含图片的特征点、特征点描述子和颜色
	 */
	public static void extractFeatures(List<Mat> imageList, List<ImageData> imageDataList) {
		for (int i = 0; i < imageList.size(); i++) {
			ImageData temp = detectFeature(imageList.get(i));
			System.out.println("第"+(i+1)+"张图像检测到"+temp.getKeyPoint().height()+"个特征点");
			if (temp.getKeyPoint().height() < 10) {// 如果一张图片检测到的特征点数小于10，则舍弃
				imageList.remove(i);
				continue;
			}
			imageDataList.add(temp);
		}
	}

	/**
	 * 特征点匹配函数
	 * 
	 * @param imageDataForAll 检测出的图片特征点描述子所在的列表
	 * @param matchesForAll	计算出的匹配点列表
	 */
	public static void matchFeatures(List<ImageData> imageDataForAll,List<MatOfDMatch> matchesForAll) {
		matchesForAll.clear();
		for (int i = 0; i < imageDataForAll.size() - 1; i++){
			MatOfDMatch matches=null;
			matches=matchFeatures(imageDataForAll.get(i), imageDataForAll.get(i+1));
			System.out.println("第"+(i+1)+"与第"+(i+2)+"张图像检测出"+matches.height()+"个匹配点对");
			matchesForAll.add(matches);
		}
	}
	
	/**
	 * 特征点匹配函数
	 * 
	 * @param query 输入的一张待匹配图片的信息
	 * @param train 输入的一张待匹配图片的信息
	 * @return 返回两张图像的匹配结果MatOfDMatch
	 */
	public static MatOfDMatch matchFeatures(ImageData query, ImageData train) {
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
		List<MatOfDMatch> knnmatches = new LinkedList<MatOfDMatch>();
		List<DMatch> matchesList = new LinkedList<DMatch>();
		matcher.knnMatch(query.getDescriptors(), train.getDescriptors(), knnmatches,2);
		
		//获取满足Ratio Test的最小匹配的距离
		double min_dist =Double.MAX_VALUE;
		for (int r = 0; r < knnmatches.size(); r++){
			//Ratio Test
			if (knnmatches.get(r).toArray()[0].distance > 0.6*knnmatches.get(r).toArray()[1].distance)
				continue;
			double dist = knnmatches.get(r).toArray()[0].distance;
			if (dist < min_dist) min_dist = dist;
		}
		matchesList.clear();
		for (int r = 0; r < knnmatches.size(); r++)
		{
			//排除不满足Ratio Test的点和匹配距离过大的点
			if (
				knnmatches.get(r).toArray()[0].distance > 0.6*knnmatches.get(r).toArray()[1].distance ||
				knnmatches.get(r).toArray()[0].distance > 5 * (min_dist>10.0f?min_dist:10.0f)
				)
				continue;
			//保存匹配点
			matchesList.add(knnmatches.get(r).toArray()[0]);
		}
		MatOfDMatch goodMatch=new MatOfDMatch();
		goodMatch.fromList(matchesList);
		return goodMatch;
	}

	/**
	 * 特征点检测函数，输入Mat，使用SIFT算法检测特征点并生成特征点描述子
	 * 
	 * @param image 输入的待检测的图片
	 * @return ImageData 输出的特征点检测结果和特征点描述子以及颜色信息
	 */
	private static ImageData detectFeature(Mat image) {
		try {
			int channels = image.channels();
			if (channels != 3) {
				throw new Exception("图像不为RGB");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		Mat img = new Mat(image.height(), image.width(), CvType.CV_8UC3); // 用于存储转化为BGR后的图像
		FeatureDetector detector = FeatureDetector.create(FeatureDetector.AKAZE);
		DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.AKAZE);
		MatOfKeyPoint keypoints = new MatOfKeyPoint();
		Mat descriptors = new Mat();
		Imgproc.cvtColor(image, img, Imgproc.COLOR_RGB2BGR, 3); // 转换为BGR彩色模式是为了效率考虑
		detector.detect(img, keypoints);
		extractor.compute(img, keypoints, descriptors);
		Mat color = extractKeypointColor(keypoints, image);
		return ImageData.newInstance(keypoints, descriptors, color);
	}

	/**
	 * 特征点颜色提取
	 * 
	 * @param keyPoint 特征点列表
	 * @param img      原图像
	 * @return 特征点颜色的Mat
	 */
	private static Mat extractKeypointColor(MatOfKeyPoint keyPoint, Mat img) {
		int channels = img.channels();
		Mat color = null;
		try {
			if (channels == 3) {
				color = new Mat(keyPoint.height(), 1, CvType.CV_32FC3);
			} else {
				throw new Exception("进行特征点检测的图像不是RGB类型");
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		float scale = 1.0f / 256.0f;
		for (int i = 0; i < keyPoint.toList().size(); i++) {
			int y = (int) keyPoint.toList().get(i).pt.y;
			int x = (int) keyPoint.toList().get(i).pt.x;
			double[] tmp = img.get(y, x);
			for (int j = 0; j < color.channels(); j++) {
				tmp[j] *= scale;
			}
			color.put(i, 0, tmp);
		}
		return color.clone();
	}

}
