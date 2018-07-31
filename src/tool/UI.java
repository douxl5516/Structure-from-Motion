package tool;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.highgui.VideoCapture;

public class UI {
	
	/**
	 * 	从视频读取固定间隔帧数的图片，并加入最后一帧，返回List<Mat>
	 * @param filePath 视频的存储路径
	 * @param frameRate 间隔多少帧取一张图像
	 * @param imageList 返回获取到的帧的列表，存入imageList 
	 */
	public static void getMatListFromVideo(String filePath,int frameRate,List<Mat> imageList,Mat lastImage){
		try {
			VideoCapture capture = new VideoCapture(filePath);// 读取视频 
			if (!capture.isOpened()) {
				throw new Exception("视频文件打开失败。");
			} else {
				Mat current_image = new Mat();
				int count = 0;
				capture.read(current_image);
				while (true) {
					capture.read(current_image);
					if (!current_image.empty()) {
						if (count % frameRate == 0) {
							imageList.add(current_image.clone());
						}
						lastImage = current_image.clone();
						count++;
					} else {
						if(count%frameRate!=1) {
							imageList.add(lastImage);
						}
						capture.release();
						break;
					}
				}
			}
		}catch (Exception e) {
			e.printStackTrace();
		}
	}
}
