package tests;

import java.util.ArrayList;

import part2.TrainingExample;

public class ArbitraryTest {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	// chainLength * possible labels
	// These two are initialized every time because of different chain lengths
	public static double[][][] computedEquivalentExponents;
	public static double[] computedEquivalentExponentsWithPredictionAsParam;

	public static int numOfAbstractLabels = 5;
	public static int numOfPredictedValues = 2;
	public static int numOfObservedValues = 2;
	
	
	public static double[][] wordIndexLabelIndexMatrix;
	public static double[][] labelIndexLabelIndexMatrix;
	public static double[][] labelIndexPredictedNodeMatrix;
	public static double[][] predictedNodeObservedNodeMatrix;

	/**
	 * Uses belief propagation to calculate log of partition function.
	 * 
	 * @param example
	 * @return
	 */
	/**
	 * Changes to include observation as a variable.
	 */
	private static double computePartitionFunctionEquivalentExponent(
			TrainingExample example) {
		ArrayList<Integer> wordIndices = example.wordIndices;
		int chainLength = wordIndices.size();
		if (chainLength == 1) {
			System.err
					.println("There is an error. No chain should be of length 1.");
		}
		int wordIndex;

		// added for computing derivatives
		// special case: last chain label with prediction node as function
		// parameter
		computedEquivalentExponents = new double[numOfPredictedValues][chainLength][numOfAbstractLabels];
		computedEquivalentExponentsWithPredictionAsParam = new double[numOfPredictedValues];

		double[] prevExponents = new double[numOfAbstractLabels];
		double[] succExponents = new double[numOfAbstractLabels];
		double[] temp = new double[numOfAbstractLabels];
		double[] chainEquivalentExponents = new double[numOfPredictedValues];
		// ****************************************************************************//
		double[] chainPredEquivalentExponents = new double[numOfPredictedValues
				* numOfObservedValues];
		// ****************************************************************************//
		double totalEquivalentExponent;

		for (int pred = 0; pred < numOfPredictedValues; pred++) {
			for (int pos = 0; pos < chainLength; pos++) {
				wordIndex = wordIndices.get(pos);
				// last label will not have label-label edge
				// first label position
				if (pos == 0) {
					for (int succ = 0; succ < numOfAbstractLabels; succ++) {
						// succExponents[succ] = 0;
						for (int prev = 0; prev < numOfAbstractLabels; prev++) {
							// prevExponents[prev] = 0;
							prevExponents[prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
									+ labelIndexPredictedNodeMatrix[prev][pred]
									+ labelIndexLabelIndexMatrix[prev][succ];
						}
						succExponents[succ] = computeEquivalentExponent(prevExponents);
						// added for computing derivative
						computedEquivalentExponents[pred][pos][succ] = succExponents[succ];
					}
				}
				// last label position
				else if (pos == chainLength - 1) {
					for (int i = 0; i < numOfAbstractLabels; i++) {
						temp[i] = succExponents[i];
					}

					for (int prev = 0; prev < numOfAbstractLabels; prev++) {
						// prevExponents[prev] = 0;
						prevExponents[prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
								+ labelIndexPredictedNodeMatrix[prev][pred]
								+ temp[prev];
					}
					chainEquivalentExponents[pred] = computeEquivalentExponent(prevExponents);
					// added for computing derivatives
					computedEquivalentExponentsWithPredictionAsParam[pred] = chainEquivalentExponents[pred];
				}
				// positions in between
				else {
					for (int i = 0; i < numOfAbstractLabels; i++) {
						temp[i] = succExponents[i];
					}

					for (int succ = 0; succ < numOfAbstractLabels; succ++) {
						// succExponents[succ] = 0;
						for (int prev = 0; prev < numOfAbstractLabels; prev++) {
							// prevExponents[prev] = 0;
							prevExponents[prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
									+ labelIndexPredictedNodeMatrix[prev][pred]
									+ labelIndexLabelIndexMatrix[prev][succ]
									+ temp[prev];
						}
						succExponents[succ] = computeEquivalentExponent(prevExponents);
						// added for computing derivatives
						computedEquivalentExponents[pred][pos][succ] = succExponents[succ];
					}
				}

			}
			// ****************************************************************************//
			// !!! modification here to include observation as a variable
			// Also reduces complexity
			for (int ob = 0; ob < numOfObservedValues; ob++) {
				chainPredEquivalentExponents[pred * numOfObservedValues + ob] = chainEquivalentExponents[pred]
						+ predictedNodeObservedNodeMatrix[pred][ob];
			}
			// chainPredEquivalentExponents[pred] =
			// chainEquivalentExponents[pred]
			// +
			// predictedNodeObservedNodeMatrix[pred][example.observedRelation];
			// ****************************************************************************//

		}
		totalEquivalentExponent = computeEquivalentExponent(chainPredEquivalentExponents);
		// System.out.println("belief propagation result: "
		// + totalEquivalentExponent);
		return totalEquivalentExponent;
	}

	private static double computeEquivalentExponent(double[] exponents) {
		long tmp = 0;
		double equivalent;
		// double max = maximum(exponents);
		double max;
		max = exponents[0];
		for (int i = 1; i < exponents.length; i++) {
			if (exponents[i] > max)
				max = exponents[i];
		}
		if (max == Double.NEGATIVE_INFINITY)
			return Double.NEGATIVE_INFINITY;
		double sum = 0;
		for (int i = 0; i < exponents.length; i++) {
			if (exponents[i] - max > -20) {
				// sum += Math.exp(exponents[i] - max);
				// sum += exp(exponents[i] - max);
				tmp = (long) (1512775 * (exponents[i] - max) + 1072632447);
				sum += Double.longBitsToDouble(tmp << 32);
			}
		}
		// equivalent = max + Math.log(sum);
		// equivalent = max + log(sum);
		equivalent = max + 6 * (sum - 1) / (sum + 1 + 4 * (Math.sqrt(sum)));
		return equivalent;
	}
	
}
