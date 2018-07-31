package type;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;

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
}