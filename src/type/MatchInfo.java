package type;

import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.features2d.DMatch;

public class MatchInfo {
    private MatOfDMatch matches;
    private Mat fm;
    public static MatchInfo newInstance(MatOfDMatch matches, Mat fm){
        MatchInfo matchInfo = new MatchInfo();
        matchInfo.matches = matches;
        matchInfo.fm = fm;
        return matchInfo;
    }
    public MatOfDMatch getMatches(){return this.matches;}
    public Mat getFM(){return this.fm;}
    public String toString () {
		StringBuilder s=new StringBuilder();
		s.append("Mat Of Matches:"+Types.NEW_LINE);
		List<DMatch>matchList= matches.toList();
		for(DMatch d:matchList) {
			s.append(d.toString()+Types.NEW_LINE);
		}
		s.append("fm:"+Types.NEW_LINE);
		s.append(fm.dump());
		return s.toString();
    }
}