package tool;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

public class Optimalize {
	public static double Sobel(Mat imageRaw) {
        Mat imageGrey=Mat.zeros(imageRaw.size(),imageRaw.type());
        Mat imageSobel=Mat.zeros(imageRaw.size(),imageRaw.type());
        Imgproc.cvtColor(imageRaw, imageGrey, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Sobel(imageGrey, imageSobel, CvType.CV_16U, 1, 1);
        return Core.mean(imageSobel).val[0];
	}
}
