package main;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;

import tool.UI;
import type.ImageData;
import type.MatchInfo;
import utilities.CameraModel;
import utilities.Reconstruction;

public class Main {

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	private static final int SKIP_FRAME_NUMBER = 30;
	private static final String VIDEO_FILE_NAME = "CVdata\\sfm\\sfm_1.mp4";
	private static final String IMG_LIST_FILE_NAME = "CVdata\\sfm\\imglist.txt";
	private static final String CALIB_LIST_FILE_NAME = "CVdata\\zhang_calib\\calibdata.txt";

	public static void main(String[] args) {
		Mat lastImage = new Mat();
		List<Mat> imageList = new ArrayList<Mat>();
		List<ImageData> imageDataList = new ArrayList<ImageData>();

		// 从视频或图像列表读取图像
		UI.getMatListFromVideo(VIDEO_FILE_NAME, SKIP_FRAME_NUMBER, imageList);
		lastImage = imageList.get(imageList.size() - 1);
		// UI.getMatListFromImgList(IMG_LIST_FILE_NAME, imageList, lastImage);

		// 展示读取出来的图片
		// for (Mat img : imageList) new ImageUI(img,"images").imshow().waitKey(0);
		// new ImageUI(lastImage,"last one").imshow().waitKey(0);

		// 相机标定
		// CameraModel cm = new CameraModel(CALIB_LIST_FILE_NAME);
		CameraModel cm=new CameraModel(new MatOfDouble(2759.48,0,1520.69,0,2764.16,1006.81,0,0,1).reshape(1,3));

		// 三维重建
		Reconstruction r = new Reconstruction(cm.getCameraMatrix());
		Reconstruction.extractFeatures(imageList, imageDataList);
		MatchInfo m = Reconstruction.matchFeatures(imageDataList.get(0), imageDataList.get(1));
		Mat cloud = r.InitPointCloud(imageDataList.get(0), imageDataList.get(1), m.getMatches(), imageList.get(1));
		System.out.println(cloud.dump());
		for(int i=1;i<imageDataList.size()-1;i++) {
			System.out.println(r.addImage(imageDataList.get(i), imageDataList.get(i+1), Reconstruction.matchFeatures(imageDataList.get(i), imageDataList.get(i+1)).getMatches(), imageList.get(i+1)).dump());
		}
		

	}

}