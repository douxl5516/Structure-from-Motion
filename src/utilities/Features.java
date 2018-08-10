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
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import type.ImageData;
import type.MatchInfo;

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
	public static void matchFeatures(List<ImageData> imageDataForAll,List<MatchInfo> matchesForAll) {
		matchesForAll.clear();
		for (int i = 0; i < imageDataForAll.size() - 1; i++){
			MatchInfo matches=null;
			matches=matchFeatures(imageDataForAll.get(i), imageDataForAll.get(i+1));
			matchesForAll.add(matches);
		}
	}
	
	/**
	 * 特征点匹配函数
	 * 
	 * @param query 输入的一张待匹配图片的信息
	 * @param train 输入的一张待匹配图片的信息
	 * @return 返回两张图像的匹配结果MatchInfo
	 */
	public static MatchInfo matchFeatures(ImageData query, ImageData train) {
		DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
		MatOfDMatch matches = new MatOfDMatch();
		matcher.match(query.getDescriptors(), train.getDescriptors(), matches);
		
		List<DMatch> matchesList = matches.toList();
		List<KeyPoint> kpList1 = query.getKeyPoint().toList();
		List<KeyPoint> kpList2 = train.getKeyPoint().toList();
		
		LinkedList<Point> points1 = new LinkedList<>();
		LinkedList<Point> points2 = new LinkedList<>();
		for (int i = 0; i < matchesList.size(); i++) {
			points1.addLast(kpList1.get(matchesList.get(i).queryIdx).pt);
			points2.addLast(kpList2.get(matchesList.get(i).trainIdx).pt);
		}
		MatOfPoint2f kp1 = new MatOfPoint2f();
		MatOfPoint2f kp2 = new MatOfPoint2f();
		kp1.fromList(points1);
		kp2.fromList(points2);
		
		Mat inliner = new Mat();
//		Mat F = Calib3d.findHomography(kp1, kp2, Calib3d.FM_RANSAC, 3, inliner, 30, 0.99); // 求解出的inliner是图片上的变换矩阵
		Mat F = Calib3d.findFundamentalMat(kp1, kp2, Calib3d.FM_RANSAC, 3, 0.99, inliner); // 求解出的inliner是基础矩阵
		System.out.println(F.dump());
		List<Byte> isInliner = new ArrayList<>();
		Converters.Mat_to_vector_uchar(inliner, isInliner);
		LinkedList<DMatch> good_matches = new LinkedList<>();
		MatOfDMatch gm = new MatOfDMatch();
		for (int i = 0; i < isInliner.size(); i++) {
			if (isInliner.get(i) != 0) {
				good_matches.addLast(matchesList.get(i));
			}
		}
		gm.fromList(good_matches);
		return MatchInfo.newInstance(gm, F);
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
		FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
		DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.ORB);
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
				throw new Exception("进行特征点检测的图像不是RGB。");
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
