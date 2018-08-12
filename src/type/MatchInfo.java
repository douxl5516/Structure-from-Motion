package type;

import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;

public class MatchInfo {
    private MatOfDMatch matches;
    private Mat fm;
    private Mat mask;
    public static MatchInfo newInstance(MatOfDMatch matches, Mat fm,Mat mask){
        MatchInfo matchInfo = new MatchInfo();
        matchInfo.matches = matches;
        matchInfo.fm = fm;
        matchInfo.mask=mask;
        return matchInfo;
    }
    public MatOfDMatch getMatches(){return this.matches;}
    public Mat getFM(){return this.fm;}
    public Mat getMask(){return this.mask;}
}
