package utilities;

import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgproc.Imgproc;

import type.ImageData;

public class Reconstruction {
	public static void extractFeatures(List<Mat> imgList, List<ImageData> imgDataList) {
		for(Mat m:imgList) {
			imgDataList.add(detectFeature(m));
		}
	}
	
	public static ImageData detectFeature(Mat mRgba){
        Mat img = new Mat(mRgba.height(), mRgba.width(), CvType.CV_8UC3);
        Imgproc.cvtColor(mRgba, img, Imgproc.COLOR_RGBA2BGR, 3);
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.SIFT);
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        Mat descriptors = new Mat();
        detector.detect(img, keypoints);
        extractor.compute(img, keypoints, descriptors);
        Mat color = extractKeypointColor(keypoints, mRgba);
        return ImageData.newInstance(keypoints,descriptors,color);
    }
    public static Mat extractKeypointColor(MatOfKeyPoint keyPoint, Mat img){
        Mat color = new Mat(keyPoint.height(),1,CvType.CV_32FC4);
        float scale = 1.0f/256.0f;
        for(int i=0;i<keyPoint.toList().size();i++){
            int y = (int)keyPoint.toList().get(i).pt.y;
            int x = (int)keyPoint.toList().get(i).pt.x;
            double[] tmp = img.get(y, x);
            for(int j=0;j<color.channels();j++){
                tmp[j] *= scale;
            }
            color.put(i,0,tmp);
        }
        return color.clone();
    }

}
