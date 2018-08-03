package tool;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.LinkedList;

import javax.imageio.ImageIO;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point3;
import org.opencv.imgcodecs.Imgcodecs;

public class Format {
	
	/**
     * BufferedImage转换成Mat
     * 
     * @param original 要转换的BufferedImage
     * @param imgType bufferedImage的类型 如 BufferedImage.TYPE_3BYTE_BGR
     * @param matType 转换成mat的type 如 CvType.CV_8UC3
     * @return 转换后的Mat
     */
	public static Mat BufferedImage2Mat (BufferedImage original, int imgType, int matType) {
        if (original == null) {
            throw new IllegalArgumentException("original == null");
        }
        // Don't convert if it already has correct type
        if (original.getType() != imgType) {
            // Create a buffered image
            BufferedImage image = new BufferedImage(original.getWidth(), original.getHeight(), imgType);
            // Draw the image onto the new buffer
            Graphics2D g = image.createGraphics();
            try {
                g.setComposite(AlphaComposite.Src);
                g.drawImage(original, 0, 0, null);
            } finally {
                g.dispose();
            }
        }
        byte[] pixels = ((DataBufferByte) original.getRaster().getDataBuffer()).getData();
        Mat mat = Mat.eye(original.getHeight(), original.getWidth(), matType);
        mat.put(0, 0, pixels);
        return mat;
    }
	
	/**
     *	重载BufferedImage转换成Mat，使用原BufferedImage的格式
     * 
     * @param original 要转换的BufferedImage
     * @param matType 转换成mat的type 如 CvType.CV_8UC3
     * @return 转换后的Mat
     */
	public static Mat BufferedImage2Mat (BufferedImage original, int matType) {
		return BufferedImage2Mat(original,original.getType(),matType);
	}
	
	/**
     * Mat转换成BufferedImage
     * 
     * @param matrix 要转换的Mat
     * @param fileExtension 格式为 ".jpg", ".png", etc
     * @return 转换后的BufferedImage
     */
    public static BufferedImage Mat2BufferedImage (Mat matrix, String fileExtension) {
        // convert the matrix into a matrix of bytes appropriate for this file extension
        MatOfByte mob = new MatOfByte();
        Imgcodecs.imencode(fileExtension, matrix, mob);
        // convert the "matrix of bytes" into a byte array
        byte[] byteArray = mob.toArray();
        BufferedImage bufImage = null;
        try {
            InputStream in = new ByteArrayInputStream(byteArray);
            bufImage = ImageIO.read(in);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return bufImage;
    }
    
    /**
     * Mat转化为MatOfPoint3f
     * @param m 待转化的Mat
     * @return 转化后的MatOfPoint3f
     */
    public static MatOfPoint3f Mat2MatOfPoint3f(Mat m) {
		MatOfPoint3f points = new MatOfPoint3f();
		LinkedList<Point3> ptlist = new LinkedList<>();
		Point3 pt = new Point3();
		Mat doubleM = new Mat();
		m.convertTo(doubleM, CvType.CV_64F);
		for (int i = 0; i < m.height(); i++) {
			double[] tmp = new double[3];
			doubleM.get(0, 0, tmp);
			pt.set(tmp);
			ptlist.add(pt);
		}
		points.fromList(ptlist);
		return points;
	}
}
