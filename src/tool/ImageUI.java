package tool;

import java.awt.Graphics;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;

import javax.swing.JFrame;
import javax.swing.JPanel;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

/**
 * 实现了类似C++中的图像显示方式
 * 
 * examples: Mat mat = Mat.eye(1000, 2000, CvType.CV_8UC3); String window_name =
 * "mat" ImageUI ig = new ImageGui(mat,window_name); ig.imshow(); ig.waitKey(0);
 */
public class ImageUI extends JPanel implements KeyListener, MouseListener {

	public ImageUI(Mat m, String window) {
		super();
		init(m, window);
	}

	// Elements for paint.
	private Mat mat;
	private Mat res = new Mat();
	private boolean firstPaint = true;
	private BufferedImage out;
	int type;
	private String WINDOW = "";
	private JFrame jframe = new JFrame();
	byte[] data;
	private Point[] p = new Point[4];
	private int count = 0;

	private void Mat2BufIm() {
		mat.get(0, 0, data);
		out.getRaster().setDataElements(0, 0, mat.cols(), mat.rows(), data);
	}

	public Mat getTargetImage() {
		System.out.println("请在图像上选出进行透视变换所用的四点");
		if (firstPaint) {
			jframe.setVisible(true);
			firstPaint = false;
		}
		Mat2BufIm();
		this.repaint();
		return mat;
	}

	private void init(Mat m, String window) {
		this.mat = m;
		data = new byte[mat.cols() * mat.rows() * (int) mat.elemSize()];

		WINDOW = window;

		if (mat.channels() == 1)
			type = BufferedImage.TYPE_BYTE_GRAY;
		else
			type = BufferedImage.TYPE_3BYTE_BGR;
		out = new BufferedImage(mat.cols(), mat.rows(), type);
		Mat2BufIm();
		jframe.add(this);
		jframe.setSize(mat.cols(), mat.rows());
		jframe.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		jframe.setTitle(WINDOW);
		jframe.addKeyListener(this);

		this.addMouseListener(this);
	}

	@Override
	public void paintComponent(Graphics g) {
		g.drawImage(out, 0, 0, null);
	}

	public ImageUI imshow() {
		if (firstPaint) {
			jframe.setVisible(true);
			firstPaint = false;
		}
		Mat2BufIm();
		this.repaint();
		return this;
	}

	// Elements for waitKey.
	private static Object mt = new Object();
	private static int lastKey = 0;
	private static int key = 0;

	public static int waitKey(int millisecond) {
		// TODO 实现监听键盘
		try {
			if (millisecond == 0) {
				synchronized (mt) {
					mt.wait();
				}
			}
			Thread.sleep(millisecond);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		int ret = -1;
		if (key != lastKey) {
			ret = key;
			lastKey = key;
		}
		return ret;
	}

	@Override
	public void keyPressed(KeyEvent e) {
		synchronized (mt) {
			mt.notifyAll();
		}
		this.key = e.getKeyCode();
	}

	@Override
	public void keyReleased(KeyEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void keyTyped(KeyEvent arg0) {
		// TODO Auto-generated method stub

	}

	@Override
	public void mouseClicked(MouseEvent e) {
		// TODO Auto-generated method stub
		System.out.println("Point " + (count + 1) + " at [" + e.getX() + "," + e.getY() + "]");
		p[count] = new Point(e.getX(), e.getY());
		count++;
		if (count == 4) {
			MatOfPoint2f src = new MatOfPoint2f(p[0], p[1], p[2], p[3]);
			MatOfPoint2f dis = new MatOfPoint2f(new Point(0, 0), new Point(mat.width(), 0),
					new Point(mat.width(), mat.height()), new Point(0, mat.height()));
			Mat transform = Imgproc.getPerspectiveTransform(src, dis);
			Imgproc.warpPerspective(mat, res, transform, mat.size());
			count %= 4;
		}
	}

	@Override
	public void mouseEntered(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	@Override
	public void mouseExited(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	@Override
	public void mousePressed(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	@Override
	public void mouseReleased(MouseEvent e) {
		// TODO Auto-generated method stub

	}

	public Mat getRes() {
		return res;
	}
}