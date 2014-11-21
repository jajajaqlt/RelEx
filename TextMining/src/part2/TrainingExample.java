package part2;

import java.util.ArrayList;

public class TrainingExample {
	public int chainLength;
	public ArrayList<Integer> wordIndices;
	public ArrayList<Integer> distinctWordIndices;
	public ArrayList<String> words;
	public ArrayList<Integer> labels;
	// for the word of disease
	public int relationWordIndex1;
	public int relationWordIndex2;
	// 0: no relation; 1: treated_by, treats, may_be_treated_by, may_treat
	public int observedRelation;
	public int predictedRelation;
	public ArrayList<DenominatorExample> denominatorExamples;
	
	public TrainingExample() {
		wordIndices = new ArrayList<Integer>();
		labels = new ArrayList<Integer>();
		words = new ArrayList<String>();
		observedRelation = 0;
		denominatorExamples = new ArrayList<DenominatorExample>();
		distinctWordIndices = new ArrayList<Integer>();
	}

	public TrainingExample getDeepCopy() {
		TrainingExample example = new TrainingExample();
		example.chainLength = this.chainLength;
		for (int i = 0; i < this.chainLength; i++) {
			example.wordIndices.add(this.wordIndices.get(i));
			example.labels.add(this.labels.get(i));
		}
		// problem here
		for (int i = 0; i < this.distinctWordIndices.size(); i++) {
			example.distinctWordIndices.add(this.distinctWordIndices.get(i));
		}
		example.observedRelation = this.observedRelation;
		example.predictedRelation = this.predictedRelation;
		return example;
	}
}
