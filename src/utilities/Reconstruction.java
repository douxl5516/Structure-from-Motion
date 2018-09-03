package utilities;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Scalar;

import tool.Format;
import tool.UI;
import type.ImageData;
import type.MatchInfo;

public class Reconstruction {

	private Mat cameraMat = new Mat(3, 3, CvType.CV_64F); // 相机内参
	private Mat pointCloud;
	private Mat color;
	private Mat LastP; // 最后一张图像的外参矩阵
	private ArrayList<int[]> correspondence_idx = new ArrayList<>();
	List<Mat> imageList;
	List<ImageData> imageDataList;
	List<MatchInfo> matchesList;
	private final float scale = 1 / 256;

	public Reconstruction(Mat cameraMat, List<Mat> imageList, List<ImageData> imageDataList,
			List<MatchInfo> matchesList) {
		this.cameraMat = cameraMat;
		this.imageList = imageList;
		this.imageDataList = imageDataList;
		this.matchesList = matchesList;
	}

	public void runSfM() {
		InitStructure(imageDataList.get(0), imageDataList.get(1), matchesList.get(0).getMatches(), imageList.get(1));
		System.out.println("点云size:" + pointCloud.size());
		for (int i = 1; i < matchesList.size(); i++) {
			addImage(imageDataList.get(i), imageDataList.get(i + 1), matchesList.get(i).getMatches(),
					imageList.get(i + 1));
			System.out.println("点云size:" + pointCloud.size());
		}
		UI.writePointCloud("output\\pointcloud.txt", pointCloud);
	}

	/**
	 * 初始化空间点云
	 * 
	 * @param left  第一幅图像的相关信息
	 * @param right 第二幅图像的相关信息
	 * @param gm    图像序列中第一张和第二张图像匹配点对DMatch列表
	 * @param img   提供点云颜色的图像
	 * @return 返回空间点云
	 */
	private Mat InitStructure(ImageData left, ImageData right, MatOfDMatch gm, Mat img) {
		color = new Mat();
		MatOfKeyPoint leftPoint = left.getKeyPoint(); // 原左图关键点mat
		MatOfKeyPoint rightPoint = right.getKeyPoint(); // 原右图关键点mat
		List<KeyPoint> leftPointList=leftPoint.toList(); // 原左图关键点列表
		List<KeyPoint> rightPointList=rightPoint.toList(); // 原右图关键点列表
		List<DMatch> matchesList=gm.toList();
		Mat rot1 = Mat.eye(3, 3, CvType.CV_64F); // 左图相机旋转矩阵
		Mat t1 = Mat.zeros(3, 1, CvType.CV_64F); // 左图相机平移矩阵
		Mat rot2 = new Mat(3, 3, CvType.CV_64F); // 右图相机旋转矩阵
		Mat t2 = new Mat(3, 1, CvType.CV_64F); // 右图相机平移矩阵
		Mat mask = new Mat();
		LinkedList<Point> ptlist1 = new LinkedList<>(); // 经匹配后的左图特征点列表
		LinkedList<Point> ptlist2 = new LinkedList<>(); // 经匹配后的右图特征点列表
		MatOfPoint2f kp1 = new MatOfPoint2f();
		MatOfPoint2f kp2 = new MatOfPoint2f();
		int[] left_idx = new int[leftPoint.height()];
		int[] right_idx = new int[rightPoint.height()];
		Arrays.fill(left_idx, -1);
		Arrays.fill(right_idx, -1);

		// 获取匹配点对并建立索引
		for (int i = 0; i < matchesList.size(); i++) {
			DMatch match = matchesList.get(i);
			ptlist1.addLast(leftPointList.get(match.queryIdx).pt);
			ptlist2.addLast(rightPointList.get(match.trainIdx).pt);
		}
		kp1.fromList(ptlist1);
		kp2.fromList(ptlist2);
		Mat E = Calib3d.findEssentialMat(kp1, kp2, cameraMat, Calib3d.RANSAC, 0.999, 1.0, mask);
		Calib3d.recoverPose(E, kp1, kp2, cameraMat, rot2, t2, mask);
		maskoutPoints(ptlist1, mask);
		maskoutPoints(ptlist2, mask);
		kp1.release();
		kp2.release();
		kp1.fromList(ptlist1);
		kp2.fromList(ptlist1);
		Mat P1 = computeProjMat(cameraMat, rot1, t1);
		Mat P2 = computeProjMat(cameraMat, rot2, t2);
		left.setProj(P1.clone());
		right.setProj(P2.clone());
		Mat pc_raw = new Mat();
		Calib3d.triangulatePoints(P1, P2, kp1, kp2, pc_raw);
		pointCloud = divideLast(pc_raw);
		LastP = P2.clone();

		int idx = 0;
		for (int i = 0; i < matchesList.size(); i++) {
			if (mask.get(i, 0)[0] == 0)
				continue;
			DMatch match = matchesList.get(i);
			left_idx[match.queryIdx] = idx;
			right_idx[match.trainIdx] = idx;
			Point pt = rightPointList.get(match.trainIdx).pt;
			double[] tmp = img.get((int) pt.y, (int) pt.x);
			tmp[0] *= scale;
			tmp[1] *= scale;
			tmp[2] *= scale;
			Mat dummy = new Mat(1, 1, CvType.CV_32FC3);
			color.push_back(dummy);
			idx++;
		}
		correspondence_idx.add(left_idx);
		correspondence_idx.add(right_idx);
		return pointCloud.clone();
	}


	/**
	 * 向点云中添加一副图片
	 * 
	 * @param left  左图相关信息
	 * @param right 右图相关信息
	 * @param gm    匹配点对
	 * @param img   取颜色的图像
	 * @return 重建后的三维点云
	 */
	private Mat addImage(ImageData left, ImageData right, MatOfDMatch matches, Mat img) {
		MatOfPoint3f pc3f = Format.Mat2MatOfPoint3f(pointCloud); // 原有的点云的Point3f
		MatOfKeyPoint leftPoint = left.getKeyPoint(); // 原有左图关键点mat
		MatOfKeyPoint rightPoint = right.getKeyPoint(); // 原有右图关键点mat
		List<KeyPoint> leftPointList = leftPoint.toList(); // 原有左图关键点列表
		List<KeyPoint> rightPointList = rightPoint.toList(); // 原有右图关键点列表
		List<DMatch> matchesList=matches.toList();
		LinkedList<Point3> objectPoints = new LinkedList<Point3>(); // mask匹配后的空间点列表
		LinkedList<Point> imagePoints = new LinkedList<Point>(); // mask匹配后的右图的关键点列表
		MatOfPoint3f opMat = new MatOfPoint3f(); // mask匹配后的空间点mat
		MatOfPoint2f ipMat = new MatOfPoint2f(); // mask匹配后的右图的关键点mat
		LinkedList<Point> ptlist1 = new LinkedList<>(); // 经匹配后的左图特征点列表
		LinkedList<Point> ptlist2 = new LinkedList<>(); // 经匹配后的右图特征点列表
		MatOfPoint2f kp1 = new MatOfPoint2f(); // 经匹配后的左图特征点mat
		MatOfPoint2f kp2 = new MatOfPoint2f(); // 经匹配后的右图特征点mat

		List<Point3> pointCloudList = pc3f.toList();// 点云列表
		List<DMatch> matchList = matches.toList();// 匹配列表
		int[] left_idx = correspondence_idx.get(correspondence_idx.size() - 1); // 最后一张图像的匹配索引
		int[] right_idx = new int[rightPoint.height()];
		Arrays.fill(right_idx, -1);

		// 获取第i幅图像中匹配点对应的三维点，以及在第i+1幅图像中对应的像素点
		for (int i = 0; i < matchList.size(); i++) {
			DMatch match = matchList.get(i);
			int cloudIdx = left_idx[match.queryIdx];
			if (cloudIdx < 0)
				continue;
			objectPoints.add(pointCloudList.get(cloudIdx));
			imagePoints.add(rightPointList.get(match.trainIdx).pt);
		}
		opMat.fromList(objectPoints);
		ipMat.fromList(imagePoints);

		Mat rotvec = new Mat(3, 1, CvType.CV_64F);
		Mat rot = new Mat(3, 3, CvType.CV_64F);
		Mat t = new Mat(3, 1, CvType.CV_64F);
		Calib3d.solvePnPRansac(opMat, ipMat, cameraMat, new MatOfDouble(), rotvec, t);
		Calib3d.Rodrigues(rotvec, rot);
		Mat P = computeProjMat(cameraMat, rot, t);
		right.setProj(P);

		// 获取匹配点对并建立索引
		for (int i = 0; i < matchesList.size(); i++) {
			DMatch match = matchesList.get(i);
			ptlist1.addLast(leftPointList.get(match.queryIdx).pt);
			ptlist2.addLast(rightPointList.get(match.trainIdx).pt);
		}
		kp1.fromList(ptlist1);
		kp2.fromList(ptlist2);
		Mat nextStructure = new Mat();
		Calib3d.triangulatePoints(LastP, P, kp1, kp2, nextStructure);
		Mat last=divideLast(nextStructure);
		
		for(int i=0;i<matchesList.size();i++) {
			DMatch match=matchesList.get(i);
			if(left_idx[match.queryIdx]>=0) {
				//若该点在空间中已经存在，则这对匹配点对应的空间点应该是同一个，索引要相同
				right_idx[match.trainIdx]=left_idx[match.queryIdx];
				continue;
			}
			//若该点在空间中已经存在，将该点加入到结构中，且这对匹配点的空间点索引都为新加入的点的索引
			pointCloud.push_back(last.row(i));
			
			Point pt = rightPointList.get(match.trainIdx).pt;
			double[] tmp = img.get((int) pt.y, (int) pt.x);
			tmp[0] *= scale;
			tmp[1] *= scale;
			tmp[2] *= scale;
			Mat dummy = new Mat(1, 1, CvType.CV_32FC3);
			color.push_back(dummy);
			left_idx[match.queryIdx]=pointCloud.height()-1;
			right_idx[match.trainIdx]=pointCloud.height()-1;
		}
		correspondence_idx.remove(correspondence_idx.size()-1);
		correspondence_idx.add(left_idx);
		correspondence_idx.add(right_idx);
		LastP=P.clone();
		return pointCloud.clone();
	}

	/**
	 * 推测相机位置
	 * 
	 * @param left    最后一张图像
	 * @param right   经过变换后的目标图像
	 * @param matches 匹配列表
	 * @param img     用于取颜色的图像
	 * @return 相机的外参矩阵
	 */
	public Mat reckon(ImageData left, ImageData right, MatOfDMatch matches, Mat img) {
		MatOfPoint3f pc3f = Format.Mat2MatOfPoint3f(pointCloud); // 原有的点云的Point3f
		MatOfKeyPoint leftPoint = left.getKeyPoint(); // 原有左图关键点mat
		MatOfKeyPoint rightPoint = right.getKeyPoint(); // 原有右图关键点mat
		List<KeyPoint> leftPointList = leftPoint.toList(); // 原有左图关键点列表
		List<KeyPoint> rightPointList = rightPoint.toList(); // 原有右图关键点列表
		List<DMatch> matchesList=matches.toList();
		LinkedList<Point3> objectPoints = new LinkedList<Point3>(); // mask匹配后的空间点列表
		LinkedList<Point> imagePoints = new LinkedList<Point>(); // mask匹配后的右图的关键点列表
		MatOfPoint3f opMat = new MatOfPoint3f(); // mask匹配后的空间点mat
		MatOfPoint2f ipMat = new MatOfPoint2f(); // mask匹配后的右图的关键点mat
		LinkedList<Point> ptlist1 = new LinkedList<>(); // 经匹配后的左图特征点列表
		LinkedList<Point> ptlist2 = new LinkedList<>(); // 经匹配后的右图特征点列表
		MatOfPoint2f kp1 = new MatOfPoint2f(); // 经匹配后的左图特征点mat
		MatOfPoint2f kp2 = new MatOfPoint2f(); // 经匹配后的右图特征点mat

		List<Point3> pointCloudList = pc3f.toList();// 点云列表
		List<DMatch> matchList = matches.toList();// 匹配列表
		int[] left_idx = correspondence_idx.get(correspondence_idx.size() - 1); // 最后一张图像的匹配索引
		int[] right_idx = new int[rightPoint.height()];
		Arrays.fill(right_idx, -1);

		// 获取第i幅图像中匹配点对应的三维点，以及在第i+1幅图像中对应的像素点
		for (int i = 0; i < matchList.size(); i++) {
			DMatch match = matchList.get(i);
			int cloudIdx = left_idx[match.queryIdx];
			if (cloudIdx < 0)
				continue;
			objectPoints.add(pointCloudList.get(cloudIdx));
			imagePoints.add(rightPointList.get(match.trainIdx).pt);
		}
		opMat.fromList(objectPoints);
		ipMat.fromList(imagePoints);

		Mat rotvec = new Mat(3, 1, CvType.CV_64F);
		Mat rot = new Mat(3, 3, CvType.CV_64F);
		Mat t = new Mat(3, 1, CvType.CV_64F);
		Calib3d.solvePnPRansac(opMat, ipMat, cameraMat, new MatOfDouble(), rotvec, t);
		Calib3d.Rodrigues(rotvec, rot);
		Mat P = computeProjMat(cameraMat, rot, t);
		right.setProj(P);
		return P;
	}

	/**
	 * 计算投影矩阵
	 * 
	 * @param K 相机内参矩阵
	 * @param R 旋转矩阵
	 * @param T 平移矩阵
	 * @return 相机投影矩阵
	 */
	private Mat computeProjMat(Mat K, Mat R, Mat T) {
		Mat Proj = new Mat(3, 4, CvType.CV_64F);
		Mat RT = new Mat();
		RT.push_back(R.t());
		RT.push_back(T.t());
		Core.gemm(K, RT.t(), 1, new Mat(), 0, Proj, 0);
		double test[] = new double[12];
		Proj.get(0, 0, test);
		return Proj;
	}

	/**
	 * 将齐次坐标转化为真正的坐标值
	 * 
	 * @param raw 齐次坐标矩阵
	 * @return 转化后的真正坐标值
	 */
	private Mat divideLast(Mat raw) {
		Mat pc = new Mat();
		for (int i = 0; i < raw.cols(); i++) {
			Mat col = new Mat(4, 1, CvType.CV_32F);
			Core.divide(raw.col(i), new Scalar(raw.col(i).get(3, 0)), col);
			pc.push_back(col.t());
		}
		return pc.colRange(0, 3);
	}

	/**
	 * 将mask相应位置为1的点筛出
	 * 
	 * @param ptlist 待筛选的点列表
	 * @param mask   mask的Mat，0代表不匹配，需要去除，1表示匹配，需要保留
	 */
	private void maskoutPoints(List<Point> ptlist, Mat mask) {
		LinkedList<Point> copy = new LinkedList<Point>();
		copy.addAll(ptlist);
		ptlist.clear();
		for (int i = 0; i < mask.height(); i++) {
			if (mask.get(i, 0)[0] > 0) {
				ptlist.add(copy.get(i));
			}
		}
	}

	public Mat getPointCloud() {
		return pointCloud;
	}
}
