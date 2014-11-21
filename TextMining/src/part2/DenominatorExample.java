package part2;

import java.util.ArrayList;

public class DenominatorExample {
	ArrayList<Edge> wordLabelEdges;
	ArrayList<Edge> labelLabelEdges;
	ArrayList<Edge> labelPredictionEdges;
	Edge predictionObservationEdge;
	double totalWeights;
//	double totalWeightsExp;
	// The percentage one stratified example's weights makes up 
	double percentage;

	public DenominatorExample() {
		wordLabelEdges = new ArrayList<Edge>();
		labelLabelEdges = new ArrayList<Edge>();
		labelPredictionEdges = new ArrayList<Edge>();
		totalWeights = 0;
//		totalWeightsExp = 0;
		percentage = 0;
	}
}
