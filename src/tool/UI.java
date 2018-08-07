package tool;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;

public class UI {

	/**
	 * 从视频读取固定间隔帧数的图片，并加入最后一帧，返回List<Mat>
	 * 
	 * @param filePath  视频的存储路径
	 * @param frameRate 间隔多少帧取一张图像
	 * @param imageList 返回获取到的帧的列表，存入imageList
	 * @param lastImage 最后一张图片
	 */
	public static void getMatListFromVideo(String filePath, int frameRate, List<Mat> imageList) {
		Mat lastImage=new Mat();
		try {
			VideoCapture capture = new VideoCapture(filePath);// 读取视频
			if (!capture.isOpened()) {
				throw new Exception("视频文件打开失败,请检查ffmpeg.dll");
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
						if (count % frameRate != 1) {
							imageList.add(lastImage);
						}
						capture.release();
						break;
					}
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * 从图像列表读取所有图片，并加入最后一张，返回List<Mat>
	 * 
	 * @param filePath  图像列表的存储路径
	 * @param imageList 返回获取到的帧的列表，存入imageList
	 * @param lastImage 最后一张图片
	 */
	public static void getMatListFromImgList(String filePath, List<Mat> imageList, Mat lastImage) {
		try {
			BufferedReader fin = new BufferedReader(new FileReader(filePath));
			while (true) {
				String img_path = fin.readLine();
				if (img_path != null) {
					Mat imageInput = Imgcodecs.imread(img_path);
					imageList.add(imageInput.clone());
				} else {
					break;
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
