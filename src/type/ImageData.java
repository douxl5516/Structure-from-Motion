package type;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

public class ImageData {
    private MatOfKeyPoint keyPoint;
    private Mat descriptors;
    private Mat color;
    private Mat proj;

	public static ImageData newInstance(MatOfKeyPoint keyPoint, Mat descriptors, Mat color){
        ImageData imageData = new ImageData();
        imageData.keyPoint = keyPoint;
        imageData.descriptors = descriptors;
        imageData.color = color;
        return imageData;
    }
    public static ImageData newInstance(MatOfKeyPoint keyPoint, Mat descriptors){
        ImageData imageData = new ImageData();
        imageData.keyPoint = keyPoint;
        imageData.descriptors = descriptors;
        return imageData;
    }
    public void setProj(Mat proj) {this.proj = proj;}
    public MatOfKeyPoint getKeyPoint(){return this.keyPoint;}
    public Mat getDescriptors(){return this.descriptors;}
    public Mat getColor(){return this.color;}
	public Mat getProj() {return this.proj;}
}
