package main;

import java.awt.EventQueue;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import tool.ImageUI;
import tool.UI;
import utilities.CameraModel;

public class Main {

	static {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
	}

	private static final int SKIP_FRAME_NUMBER = 30;
	private static final String VIDEO_FILE_NAME = "L:\\workspaces\\CVdata\\sfm_1.mp4";
	private static final String CALIB_LIST_FILE_NAME = "L:\\workspaces\\CVdata\\zhang_calib\\calibdata.txt";

	public static void main(String[] args) {
		Mat lastImage;
		List<Mat> imageList = new ArrayList<Mat>();

		UI.getMatListFromVideo(VIDEO_FILE_NAME, SKIP_FRAME_NUMBER, imageList);
		lastImage = imageList.get(imageList.size() - 1);

		CameraModel cm=new CameraModel(CALIB_LIST_FILE_NAME);
		
		/*
		展示读取出来的图片
		for (Mat m : imageList) {
			System.out.println(m.size());
			ImageUI i = new ImageUI(m, "1");
			i.imshow();
			i.waitKey(0);
		}
		ImageUI s = new ImageUI(lastImage, "last");
		s.imshow();
		s.waitKey(0);*/
	}

}