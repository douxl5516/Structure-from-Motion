package utilities;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import tool.ImageUI;

import org.opencv.calib3d.Calib3d;

public class CameraModel {

	BufferedReader fin;
	BufferedWriter fout;
	int image_count = 0; // 图像数量
	Size image_size; // 图像尺寸
	Size board_size; // 标定板上每行每列的角点数
	MatOfPoint2f image_points_buf; // 缓存每幅图像上检测到的角点
	List<Mat> image_points_seq; // 保存检测到的所有角点
	int count; // 保存检测到的角点个数
	String result_path; // 计算结果存储的路径

	/**
	 * 构造函数，初始化相机标定参数
	 * 
	 * @param datasetPath 相机标定数据集列表文件路径
	 */
	public CameraModel(String dataset_path) {

		image_size = new Size();
		board_size = new Size(9, 6);
		image_points_buf = new MatOfPoint2f();
		image_points_seq = new ArrayList<Mat>();
		result_path = dataset_path + "/../calibrate_result.txt";

		try {
			fin = new BufferedReader(new FileReader(dataset_path));
			fout = new BufferedWriter(new FileWriter(result_path));
			while (true) {
				String img_path = fin.readLine();
				if (img_path != null) {
					image_count++;
					Mat imageInput = Highgui.imread(img_path);

					// 读入第一张图片时获取图像宽高信息
					if (image_count == 1) {
						image_size.width = imageInput.cols();
						image_size.height = imageInput.rows();
						System.out.println("图像宽度：" + image_size.width);
						System.out.println("图像高度：" + image_size.height);
					}

					// 提取角点
					if (Calib3d.findChessboardCorners(imageInput, board_size, image_points_buf) == false) {
						throw new Exception("can not find chessboard corners!");
					} else {
						Mat view_gray = new Mat();
						Imgproc.cvtColor(imageInput, view_gray, Imgproc.COLOR_BGR2GRAY);

						// 亚像素精确化
						// Size(5,5)是搜索窗口的大小,Size(-1,-1)表示没有死区
						// 第四个参数定义求角点的迭代过程的终止条件，可以为迭代次数和角点精度两者的组合
						Imgproc.cornerSubPix(view_gray, image_points_buf, new Size(5, 5), new Size(-1, -1),
								new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 30, 0.1));
						image_points_seq.add(image_points_buf);

						// 用于绘制被成功标定的角点，输入8位灰度或者彩色图像
						// 第四个参数是标志位，用来指示定义的棋盘内角点是否被完整的探测到
						// false表示有未被探测到的内角点，这时候函数会以圆圈标记出检测到的内角点
						Calib3d.drawChessboardCorners(view_gray, board_size, image_points_buf, false);

						// 展示棋盘格角点搜索结果
						// new ImageUI(view_gray, "Camera Calibration").imshow().waitKey(500);
					}
				} else {
					break;
				}
			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		int total = image_points_seq.size();
		System.out.println("图像数量：" + total);
		int CornerNum = (int) (board_size.width * board_size.height);	//每张图片上总的角点数
		System.out.println("角点提取完成");
		
		// 开始标定
		Size square_size = new Size(10, 10);	// 实际测量得到的标定板上每个棋盘格的大小
		List<Mat> object_points=new ArrayList<Mat>();	// 保存标定板上角点的三维坐标
		// 内外参数
		Mat cameraMatrix = new Mat(3, 3, CvType.CV_32FC1, Scalar.all(0));	// 摄像机内参数矩阵
		List<Integer> point_counts=new ArrayList<Integer>();  // 每幅图像中角点的数量  
		Mat distCoeffs = new Mat(1, 5, CvType.CV_32FC1, Scalar.all(0));	// 摄像机的5个畸变系数：k1,k2,p1,p2,k3
		List<Mat> tvecsMat=new ArrayList<Mat>();	// 每幅图像的旋转向量
		List<Mat> rvecsMat=new ArrayList<Mat>();	// 每幅图像的平移向量
		// 初始化标定板上角点的三维坐标
		for (int t = 0; t<image_count; t++)
		{
			List<Point3> tempPointSet=new ArrayList<Point3>();
			for (int i = 0; i<board_size.height; i++)
			{
				for (int j = 0; j<board_size.width; j++)
				{
					Point3 realPoint = new Point3();
					/* 假设标定板放在世界坐标系中z=0的平面上 */
					realPoint.x = i * square_size.width;
					realPoint.y = j * square_size.height;
					realPoint.z = 0;
					tempPointSet.add(realPoint);
				}
			}
			MatOfPoint3f tmpMat=new MatOfPoint3f();
			tmpMat.fromList(tempPointSet);
			object_points.add(tmpMat);
		}
		
		// 初始化每幅图像中的角点数量，假定每幅图像中都可以看到完整的标定板
		for (int i = 0; i<image_count; i++){
			point_counts.add(new Integer((int) (board_size.width*board_size.height)));
		}
		
		// 开始标定
		Calib3d.calibrateCamera(object_points, image_points_seq, image_size, cameraMatrix, distCoeffs, rvecsMat, tvecsMat, 0);
		System.out.println("标定完成");
		
		//保存定标结果      
		Mat rotation_matrix = new Mat(3, 3, CvType.CV_32FC1, Scalar.all(0));	//保存每幅图像的旋转矩阵
		try {
			fout.write("相机内参数矩阵："+System.getProperty("line.separator"));
			fout.write(cameraMatrix.dump()+System.getProperty("line.separator"));
			fout.write("畸变系数："+System.getProperty("line.separator"));
			fout.write(distCoeffs.dump()+System.getProperty("line.separator"));
			for (int i = 0; i<image_count; i++)
			{
				fout.write("第"+(i + 1)+"幅图像的旋转向量："+System.getProperty("line.separator"));
				fout.write(tvecsMat.get(i).dump()+System.getProperty("line.separator"));
				// 将旋转向量转换为相对应的旋转矩阵
				Calib3d.Rodrigues(tvecsMat.get(i), rotation_matrix);
				fout.write("第"+(i + 1) +"幅图像的旋转矩阵："+System.getProperty("line.separator"));
				fout.write(rotation_matrix.dump()+System.getProperty("line.separator"));
				fout.write("第"+(i + 1)+"幅图像的平移向量："+System.getProperty("line.separator"));
				fout.write(rvecsMat.get(i).dump()+System.getProperty("line.separator"));
			}
			System.out.println("标定结果已存储");
			fin.close();
			fout.flush();
			fout.close();
		}catch(IOException e) {
			e.printStackTrace();
		}
	}
}
