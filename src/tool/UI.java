package tool;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.video.Video;
import org.opencv.videoio.VideoCapture;

public class UI {

	/**
	 * 从视频读取固定间隔帧数的图片，返回List<Mat>
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
	 * 从视频读取经过筛选后清晰度最高的图片，返回List<Mat>
	 * 
	 * @param filePath 视频文件存放路径
	 * @param frameRate 每隔多少帧开始选取
	 * @param sampleSize 选取照片的样本数量
	 * @param imageList 返回的图像列表
	 */
	public static void getOptimalizedMatListFromVideo(String filePath, int frameRate,int sampleSize, List<Mat> imageList) {
		if(sampleSize>frameRate) {
			sampleSize=frameRate;
		}
		Mat lastImage=new Mat();
		try {
			VideoCapture capture = new VideoCapture(filePath);// 读取视频
			if (!capture.isOpened()) {
				throw new Exception("视频文件打开失败,请检查ffmpeg.dll");
			} else {
				int count=0;
				Mat frame = new Mat();
				List<ArrayList<Mat>> bufImgList=new ArrayList<ArrayList<Mat>>();
				while (true) {
					capture.read(frame);
					L1:if (!frame.empty()) {
						lastImage=frame.clone();
						if (count % frameRate == 0) {
							ArrayList<Mat> list=new ArrayList<Mat>();
							for(int i=0;i<sampleSize;i++) {
								list.add(frame);
								capture.read(frame);
								if(frame.empty()) {
									bufImgList.add(list);
									break L1;
								}
								lastImage=frame.clone();
							}
							bufImgList.add(list);
						}
					} else {
						for(int i=0;i<bufImgList.size();i++) {
							ArrayList<Mat> temp=bufImgList.get(i);
							double maxSobel=0;
							int index=0;
							for(int j=0;j<temp.size();j++) {
								double s=Optimalize.Sobel(temp.get(i));
								if(s>maxSobel) {
									index=j;
									maxSobel=s;
								}
							}
							imageList.add(temp.get(index));
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
	 */
	public static void getMatListFromImgList(String filePath, List<Mat> imageList) {
		try {
			BufferedReader fin = new BufferedReader(new FileReader(filePath));
			while (true) {
				String img_path = fin.readLine();
				if (img_path != null&&img_path!="") {
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
	
	/**
	 * 将三维重建的点云结果写出到文件
	 * 
	 * @param filePath  点云文件的存储路径
	 * @param pointCloud 点云列表
	 */
	public static void writePointCloud(String filePath,Mat pointCloud) {
		try {
			BufferedWriter fout = new BufferedWriter(new FileWriter(filePath));
			fout.write(pointCloud.dump());
			fout.flush();
			fout.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
