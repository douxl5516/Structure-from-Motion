package type;

import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;

public class ImageData {
    private MatOfKeyPoint keyPoint;
    private Mat descriptors;
    private Mat color;
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
    public MatOfKeyPoint getKeyPoint(){return this.keyPoint;}
    public Mat getDescriptors(){return this.descriptors;}
    public Mat getColor(){return this.color;}
}
