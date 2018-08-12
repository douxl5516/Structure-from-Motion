package utilities;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
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
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;

import tool.Format;
import type.ImageData;
import type.MatchInfo;

public class Reconstruction {

	private Mat cameraMat = new Mat(3, 3, CvType.CV_64F); // 相机内参
	private Mat pointCloud;
	private Mat color;
	private Mat LastP; // 最后一张图像的外参矩阵
	private ArrayList<int[]> correspondence_idx = new ArrayList<>();

	private final float scale = 1 / 256;

	public Reconstruction(Mat cameraMat, List<Mat> imageList, List<ImageData> imageDataList,
			List<MatchInfo> matchesList) {
		this.cameraMat = cameraMat;

		InitStructure(imageDataList.get(0), imageDataList.get(1), matchesList.get(0).getMatches(), imageList.get(1));
		System.out.println("点云size:" + pointCloud.size());
		for (int i = 1; i < matchesList.size(); i++) {
			addImage(imageDataList.get(i), imageDataList.get(i + 1), matchesList.get(i).getMatches(),
					imageList.get(i + 1));
			System.out.println("点云size:" + pointCloud.size());
		}
//		System.out.println(pointCloud.dump());
	}

	/**
	 * 初始化空间点云
	 * 
	 * @param left  第一幅图像的相关信息
	 * @param right 第二幅图像的相关信息
	 * @param gm    匹配点对DMatch列表
	 * @param img   提供点云颜色的图像
	 * @return 返回空间点云
	 */
	public Mat InitStructure(ImageData left, ImageData right, MatOfDMatch gm, Mat img) {
		color = new Mat(gm.toList().size(), 1, CvType.CV_32FC3);
		MatOfKeyPoint leftPoint = left.getKeyPoint(); // 原左图关键点列表
		MatOfKeyPoint rightPoint = right.getKeyPoint(); // 原右图关键点列表
		Mat rot1 = Mat.eye(3, 3, CvType.CV_64F); // 左图相机旋转矩阵
		Mat t1 = Mat.zeros(3, 1, CvType.CV_64F); // 左图相机平移矩阵
		Mat rot2 = new Mat(3, 3, CvType.CV_64F); // 右图相机旋转矩阵
		Mat t2 = new Mat(3, 1, CvType.CV_64F); // 右图相机平移矩阵

		LinkedList<Point> ptlist1 = new LinkedList<>(); // 经匹配后的左图特征点列表
		LinkedList<Point> ptlist2 = new LinkedList<>(); // 经匹配后的右图特征点列表
		MatOfPoint2f kp1 = new MatOfPoint2f();
		MatOfPoint2f kp2 = new MatOfPoint2f();
		int[] left_idx = new int[leftPoint.height()];
		int[] right_idx = new int[rightPoint.height()];
		Arrays.fill(left_idx, -1);
		Arrays.fill(right_idx, -1);

		// 获取匹配点对并建立索引
		for (int i = 0; i < gm.toList().size(); i++) {
			DMatch match = gm.toList().get(i);
			ptlist1.addLast(leftPoint.toList().get(match.queryIdx).pt);
			ptlist2.addLast(rightPoint.toList().get(match.trainIdx).pt);
			left_idx[match.queryIdx] = i;
			right_idx[match.trainIdx] = i;

			// 取第i对匹配点中右图的点坐标作为颜色的取值
			Point pt = rightPoint.toList().get(match.trainIdx).pt;
			double[] tmp = img.get((int) pt.y, (int) pt.x);
			tmp[0] *= scale;
			tmp[1] *= scale;
			tmp[2] *= scale;
			color.put(i, 0, tmp);
		}
		correspondence_idx.add(left_idx);
		correspondence_idx.add(right_idx);
		kp1.fromList(ptlist1);
		kp2.fromList(ptlist2);

		Mat E = Calib3d.findEssentialMat(kp1, kp2, cameraMat);
		Calib3d.recoverPose(E, kp1, kp2, cameraMat, rot2, t2);

		kp1.release();
		kp2.release();
		kp1.fromList(ptlist1);
		kp2.fromList(ptlist1);

		Mat P1 = computeProjMat(cameraMat, rot1, t1);
		Mat P2 = computeProjMat(cameraMat, rot2, t2);
		Mat pc_raw = new Mat();
		System.out.println(rot2.dump());
		System.out.println(t2.dump());

		Calib3d.triangulatePoints(P1, P2, kp1, kp2, pc_raw);
		pointCloud = divideLast(pc_raw);
		LastP = P2.clone();
		return pointCloud.clone();
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
	 * 向点云中添加一副图片
	 * 
	 * @param left
	 * @param right
	 * @param gm
	 * @param img
	 * @return
	 */
	public Mat addImage(ImageData left, ImageData right, MatOfDMatch gm, Mat img) {
		MatOfPoint3f pc3f = Format.Mat2MatOfPoint3f(pointCloud); // 点云Point3f
		MatOfKeyPoint leftPoint = left.getKeyPoint(); // 左图原关键点列表
		MatOfKeyPoint rightPoint = right.getKeyPoint(); // 右图原关键点列表
		LinkedList<Point3> pclist = new LinkedList<>();
		LinkedList<Point> right_inPC = new LinkedList<>();
		LinkedList<Point> leftlist = new LinkedList<>();
		LinkedList<Point> rightist = new LinkedList<>();
		MatOfPoint3f pc = new MatOfPoint3f();
		MatOfPoint2f kp1 = new MatOfPoint2f();
		MatOfPoint2f kp2 = new MatOfPoint2f();
		int count = pointCloud.height(); // 点云点数
		int[] left_idx = correspondence_idx.get(correspondence_idx.size() - 1);
		int[] right_idx = new int[rightPoint.height()];
		Arrays.fill(right_idx, -1);
		for (int i = 0; i < gm.toList().size(); i++) {
			if (left_idx[gm.toList().get(i).queryIdx] >= 0) {
				pclist.addLast(pc3f.toList().get(left_idx[gm.toList().get(i).queryIdx]));
				right_inPC.addLast(rightPoint.toList().get(gm.toList().get(i).trainIdx).pt);
				right_idx[gm.toList().get(i).trainIdx] = left_idx[gm.toList().get(i).queryIdx];
			} else {
				leftlist.addLast(leftPoint.toList().get(gm.toList().get(i).queryIdx).pt);
				rightist.addLast(rightPoint.toList().get(gm.toList().get(i).trainIdx).pt);
				left_idx[gm.toList().get(i).queryIdx] = count;
				right_idx[gm.toList().get(i).trainIdx] = count;
				int y = (int) rightPoint.toList().get(gm.toList().get(i).trainIdx).pt.y;
				int x = (int) rightPoint.toList().get(gm.toList().get(i).trainIdx).pt.x;
				double[] tmp = img.get(y, x);
				tmp[0] *= scale;
				tmp[1] *= scale;
				tmp[2] *= scale;
				Mat dummy = new Mat(1, 1, CvType.CV_32FC3);
				color.push_back(dummy);
				count++;
			}
		}
		pc.fromList(pclist);
		kp2.fromList(right_inPC);
		Mat rotvec = new Mat(3, 1, CvType.CV_64F);
		Mat rot = new Mat(3, 3, CvType.CV_64F);
		Mat t = new Mat(3, 1, CvType.CV_64F);
		Calib3d.solvePnPRansac(pc, kp2, cameraMat, new MatOfDouble(), rotvec, t);
		kp1.fromList(leftlist);
		kp2.fromList(rightist);
		Calib3d.Rodrigues(rotvec, rot);
		Mat P = computeProjMat(cameraMat, rot, t);
		Mat pc_raw = new Mat();
		Calib3d.triangulatePoints(LastP, P, kp1, kp2, pc_raw);

		System.out.println(pc_raw.dump());

		Mat new_PC = divideLast(pc_raw);
		pointCloud.push_back(new_PC);
		LastP = P.clone();
		return pointCloud.clone();
	}

	void reconstruct(Mat K, Mat R, Mat T, MatOfPoint2f p1, MatOfPoint2f p2, Mat structure) {
		// 两个相机的投影矩阵[R T]，triangulatePoints只支持float型
		Mat proj1 = new Mat(3, 4, CvType.CV_32FC1);
		Mat proj2 = new Mat(3, 4, CvType.CV_32FC1);

		proj1.colRange(0, 3).setTo(Mat.eye(3, 3, CvType.CV_32FC1));
		proj1.col(3).setTo(Mat.zeros(3, 1, CvType.CV_32FC1));

		R.convertTo(proj2.colRange(0, 3), CvType.CV_32FC1);
		T.convertTo(proj2.col(3), CvType.CV_32FC1);

		Mat fK = new Mat();
		K.convertTo(fK, CvType.CV_32FC1);
		proj1 = fK.mul(proj1);
		proj2 = fK.mul(proj2);

		// 三角化重建
		Calib3d.triangulatePoints(proj1, proj2, p1, p2, structure);
	}

	public Mat getPointCloud() {
		return pointCloud;
	}

}
