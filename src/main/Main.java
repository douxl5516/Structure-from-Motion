package main;

import java.util.LinkedList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;

import tool.ImageUI;
import tool.UI;
import type.ImageData;
import type.MatchInfo;
import utilities.CameraModel;
import utilities.Features;
import utilities.Reconstruction;

public class Main {
	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	private static final int SKIP_FRAME_NUMBER = 30;
	private static final String VIDEO_FILE_NAME = "CVdata\\sfm_1.mp4";
	private static final int SAMPLE_SIZE=5;
//	private static final String IMG_LIST_FILE_NAME = "CVdata\\sfm_fountain\\imglist.txt";
//	private static final String IMG_LIST_FILE_NAME = "CVdata\\sfm_entry\\imglist.txt";
//	private static final String IMG_LIST_FILE_NAME = "CVdata\\sift\\imglist.txt";
//	private static final String IMG_LIST_FILE_NAME = "CVdata\\sfm_dom\\imglist.txt";
	private static final String IMG_LIST_FILE_NAME = "CVdata\\sfm_avatar\\imglist.txt";
//	private static final String CALIB_LIST_FILE_NAME = "CVdata\\zhang_calib\\calibdata.txt";

	public static void main(String[] args) {

		Mat lastImage = new Mat();
		Mat perspectiveImage = new Mat();
		List<Mat> imageList = new LinkedList<Mat>();
		List<ImageData> imageDataList = new LinkedList<ImageData>();
		List<MatchInfo> matchesList = new LinkedList<MatchInfo>();

		// 从视频或图像列表读取图像
//		UI.getOptimalizedMatListFromVideo(VIDEO_FILE_NAME, SKIP_FRAME_NUMBER, SAMPLE_SIZE, imageList);
		UI.getMatListFromImgList(IMG_LIST_FILE_NAME, imageList);
		lastImage = imageList.get(imageList.size() - 1);
		// 展示读取出来的图片
//		 for (Mat img : imageList) new ImageUI(img,"images").imshow().waitKey(0);
//		 new ImageUI(lastImage,"last one").imshow().waitKey(0);

		// 相机标定
//		CameraModel cm = new CameraModel(CALIB_LIST_FILE_NAME);
//		CameraModel cm = new CameraModel(new MatOfDouble(2759.48, 0, 1520.69, 0, 2764.16, 1006.81, 0, 0, 1).reshape(1, 3));
		CameraModel cm = new CameraModel(new MatOfDouble(2500, 0, lastImage.cols()/2, 0, 2500, lastImage.rows()/2, 0, 0, 1).reshape(1, 3));

		// 特征点检测
		Features.extractFeatures(imageList, imageDataList);

		// 存储检测结果
		for (int i = 0; i < imageDataList.size(); i++) {
			Mat out = new Mat();
			Features2d.drawKeypoints(imageList.get(i), imageDataList.get(i).getKeyPoint(), out);
			Imgcodecs.imwrite("output\\result_of_features_" + i + ".jpg", out);
		}

		// 特征点匹配
		Features.matchFeatures(imageDataList, matchesList);
		// 存储匹配结果
		for (int i = 0; i < matchesList.size(); i++) {
			Mat outImg = new Mat();
			Features2d.drawMatches(imageList.get(i), imageDataList.get(i).getKeyPoint(), imageList.get(i + 1),imageDataList.get(i + 1).getKeyPoint(), matchesList.get(i).getMatches(), outImg);
			Imgcodecs.imwrite("output\\result_of_matches_" + i + ".jpg", outImg);
		}

		// 三维重建
		Reconstruction r = new Reconstruction(cm.getCameraMatrix(), imageList, imageDataList, matchesList);
		
		r.runSfM();

//		ImageUI transform = new ImageUI(lastImage, "图像变换");
//		transform.getTargetImage();
//		while (perspectiveImage.width() == 0) {
//			perspectiveImage = transform.getRes().clone();
//		}

//		Features.addFeatures(perspectiveImage, imageList, imageDataList, matchesList);
//		Mat P = r.reckon(imageDataList.get(imageDataList.size() - 2), imageDataList.get(imageDataList.size() - 1),matchesList.get(matchesList.size() - 1).getMatches(), imageList.get(imageList.size() - 1));
//		System.out.println("想要达到目标图像，相机外参需要为：" + P.dump());

	}
}