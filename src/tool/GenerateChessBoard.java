package tool;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

public class GenerateChessBoard {
	public static void generateChessBoard() {
		int perBoardPixel = 100;
		Size boardSize=new Size(8,6);
		Size resolution=new Size(1200,800);
		int basisHeight = (int) ((resolution.height - perBoardPixel*boardSize.height) / 2);
		int basisWidth = (int) ((resolution.width - perBoardPixel*boardSize.width) / 2);
		
		if( basisHeight < 0  ||  basisWidth < 0){
			System.out.println("Resolution doesn't match!");
		}
		
		Mat image=new Mat(resolution,CvType.CV_8UC1,Scalar.all(255));
		
	    for(int j = 0;j < boardSize.height;j++)
	    {
	        for(int i = 0;i < boardSize.width;i++)
	        {
	            int flag = (i+j) % 2;
	            if(flag == 0)
	            {
	                for(int n = j * perBoardPixel;n < (j+1) * perBoardPixel;n++)
	                    for(int m = i * perBoardPixel;m < (i+1) * perBoardPixel;m++)
	                        image.put(n + basisHeight,m + basisWidth,0);
	            }
	        }
	    }
	    new ImageUI(image,"abc").imshow();
	}
}
