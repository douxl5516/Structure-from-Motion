package tool;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.InputStream;

import javax.imageio.ImageIO;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.highgui.Highgui;

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
        Highgui.imencode(fileExtension, matrix, mob);
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
}
