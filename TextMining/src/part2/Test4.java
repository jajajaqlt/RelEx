package part2;

import java.util.ArrayList;
import java.util.Scanner;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.distribution.GammaDistribution;

public class Test4 {

	// chainLength * possible labels
	// These two are initialized every time because of different chain lengths
	public static double[][][] computedEquivalentExponents;
	public static double[] computedEquivalentExponentsWithPredictionAsParam;

	public static double[][] wordIndexLabelIndexMatrix,
			labelIndexPredictedNodeMatrix, labelIndexLabelIndexMatrix,
			predictedNodeObservedNodeMatrix;
	public static int numOfAbstractLabels = 2, numOfObservedValues = 2,
			numOfPredictedValues = 2;

	public static void main(String[] args) throws Exception {

		boolean uniformWeight = true;
		wordIndexLabelIndexMatrix = makeMatrixWithInitialValues(false, 2, 2,
				uniformWeight);
		labelIndexPredictedNodeMatrix = makeMatrixWithInitialValues(false, 2,
				2, uniformWeight);
		labelIndexLabelIndexMatrix = makeMatrixWithInitialValues(false, 2, 2,
				uniformWeight);
		predictedNodeObservedNodeMatrix = makeMatrixWithInitialValues(false, 2,
				2, uniformWeight);

		TrainingExample ex = new TrainingExample();
		int length = 3;

		for (int i = 0; i < length; i++) {
			ex.wordIndices.add(0);
			ex.labels.add(0);
		}

		testComputePartitionFunctionEquivalentExponent(ex);
		computePartitionFunctionEquivalentExponent(ex);

		String flag = "po";
		int index1 = 0, index2 = 0;
		// ex.observedRelation = 0;
		// computePartitionFunctionEquivalentExponent(ex);
		System.out
				.println(computePartitionFunctionDerivativeEquivalentExponent(
						ex, flag, index1, index2));
		testComputePartitionFunctionDerivativeEquivalentExponent(ex, flag,
				index1, index2);

		// length = 3;
		// // ex.chainLength = length;
		// ex.wordIndices.clear();
		// ex.labels.clear();
		// for (int i = 0; i < length; i++) {
		// ex.wordIndices.add(0);
		// ex.labels.add(0);
		// }
		// System.out.println(getLogLikelihoodOfLabeledTrainingExample(ex));

		// computePartitionFunctionEquivalentExponent(ex);
		// System.out
		// .println(computePartitionFunctionDerivativeEquivalentExponent(
		// ex, flag, index1, index2));
		// testComputePartitionFunctionDerivativeEquivalentExponent(ex, flag,
		// index1, index2);
		// test on
	}

	/**
	 * changes to add observation as a variable
	 */
	public static void testComputePartitionFunctionEquivalentExponent(
			TrainingExample example) {
		// brute force here to compare, set chain length 3
		ArrayList<Integer> wordIndices = example.wordIndices;
		double total = 0;
		int w1 = wordIndices.get(0), w2 = wordIndices.get(1), w3 = wordIndices
				.get(2);
		for (int ob = 0; ob < numOfObservedValues; ob++) {
			for (int pred = 0; pred < numOfPredictedValues; pred++) {
				// outermost 1
				for (int o1 = 0; o1 < numOfAbstractLabels; o1++) {
					for (int o2 = 0; o2 < numOfAbstractLabels; o2++) {
						for (int o3 = 0; o3 < numOfAbstractLabels; o3++) {
							total += Math
									.exp(wordIndexLabelIndexMatrix[w1][o1]
											+ wordIndexLabelIndexMatrix[w2][o2]
											+ wordIndexLabelIndexMatrix[w3][o3]
											+ labelIndexLabelIndexMatrix[o1][o2]
											+ labelIndexLabelIndexMatrix[o2][o3]
											+ labelIndexPredictedNodeMatrix[o1][pred]
											+ labelIndexPredictedNodeMatrix[o2][pred]
											+ labelIndexPredictedNodeMatrix[o3][pred]
											+ predictedNodeObservedNodeMatrix[pred][ob]);
						}
					}
				}
			}
		}
		double bruteForceEquivalent = Math.log(total);
		System.out.println("brute force result: " + bruteForceEquivalent);
	}

	/**
	 * This is the real test compute partition function derivative function.
	 */
	/**
	 * Changes to include observation as a variable.
	 */
	public static void testComputePartitionFunctionDerivativeEquivalentExponent(
			TrainingExample example, String flag, int index1, int index2) {
		// brute force here to compare, set chain length 3
		ArrayList<Integer> wordIndices = example.wordIndices;
		double total = 0;
		int count;
		int w1 = wordIndices.get(0), w2 = wordIndices.get(1), w3 = wordIndices
				.get(2);
		for (int ob = 0; ob < numOfObservedValues; ob++) {
			for (int pred = 0; pred < numOfPredictedValues; pred++) {
				// outermost 1
				for (int o1 = 0; o1 < numOfAbstractLabels; o1++) {
					for (int o2 = 0; o2 < numOfAbstractLabels; o2++) {
						for (int o3 = 0; o3 < numOfAbstractLabels; o3++) {
							count = 0;
							if (flag.equals("wl")) {
								if (w1 == index1 && o1 == index2)
									count++;
								if (w2 == index1 && o2 == index2)
									count++;
								if (w3 == index1 && o3 == index2)
									count++;
							} else if (flag.equals("ll")) {
								if (o1 == index1 && o2 == index2)
									count++;
								if (o2 == index1 && o3 == index2)
									count++;
							} else if (flag.equals("lp")) {
								if (o1 == index1 && pred == index2)
									count++;
								if (o2 == index1 && pred == index2)
									count++;
								if (o3 == index1 && pred == index2)
									count++;
							} else if (flag.equals("po")) {
								// if (pred == index1
								// && example.observedRelation == index2)
								// count++;
								if (pred == index1 && ob == index2)
									count++;
							}
							total += Math
									.exp(wordIndexLabelIndexMatrix[w1][o1]
											+ wordIndexLabelIndexMatrix[w2][o2]
											+ wordIndexLabelIndexMatrix[w3][o3]
											+ labelIndexLabelIndexMatrix[o1][o2]
											+ labelIndexLabelIndexMatrix[o2][o3]
											+ labelIndexPredictedNodeMatrix[o1][pred]
											+ labelIndexPredictedNodeMatrix[o2][pred]
											+ labelIndexPredictedNodeMatrix[o3][pred]
											+ predictedNodeObservedNodeMatrix[pred][ob])
									* count;
						}
					}
				}
			}
		}

		double bruteForceEquivalent = Math.log(total);
		System.out.println("derivative brute force result: "
				+ bruteForceEquivalent);
	}

	private static double computeEquivalentExponent(double[] exponents) {
		double equivalent;
		double max = maximum(exponents);
		if (max == Double.NEGATIVE_INFINITY)
			return Double.NEGATIVE_INFINITY;
		double sum = 0;
		for (int i = 0; i < exponents.length; i++) {
			sum += Math.exp(exponents[i] - max);
		}
		equivalent = max + Math.log(sum);
		return equivalent;
	}

	private static double maximum(double[] xs) {
		double max;
		max = xs[0];
		for (int i = 1; i < xs.length; i++) {
			if (xs[i] > max)
				max = xs[i];
		}
		return max;
	}

	private static double[][] makeMatrixWithInitialValues(boolean isSymmetric,
			int d1, int d2, boolean isEquallyWeighted) {
		// isEquallyWeighted = true;
		int uniformVal = 1;
		// int[] singletons = { 7, 2 };
		// double[] probabilities = { 0.5, 0.5 };
		// EnumeratedIntegerDistribution dist = new
		// EnumeratedIntegerDistribution(
		// singletons, probabilities);
		// TODO Auto-generated method stub
		double[][] matrix = new double[d1][d2];
		GammaDistribution dist = new GammaDistribution(1, 1);
		if (!isSymmetric) {
			for (int i = 0; i < d1; i++) {
				for (int j = 0; j < d2; j++) {
					if (isEquallyWeighted)
						matrix[i][j] = uniformVal;
					else
						matrix[i][j] = dist.sample();
				}
			}
		} else {
			for (int i = 0; i < d1; i++) {
				for (int j = 0; j <= i; j++) {
					// label label symmetry
					if (isEquallyWeighted)
						matrix[i][j] = uniformVal;
					else
						matrix[i][j] = dist.sample();
					matrix[j][i] = matrix[i][j];
				}
			}
		}

		return matrix;
	}

	/**
	 * Having a chain length of 2.
	 */
	// deprecated
	public static void testComputePartitionFunctionDerivativeEquivalentExponent2(
			TrainingExample example, String flag, int index1, int index2) {
		// brute force here to compare, set chain length 2
		ArrayList<Integer> wordIndices = example.wordIndices;
		double total = 0;
		int count;
		int w1 = wordIndices.get(0), w2 = wordIndices.get(1);
		for (int pred = 0; pred < numOfPredictedValues; pred++) {
			// outermost 1
			for (int o1 = 0; o1 < numOfAbstractLabels; o1++) {
				for (int o2 = 0; o2 < numOfAbstractLabels; o2++) {

					count = 0;
					if (flag.equals("wl")) {
						if (w1 == index1 && o1 == index2)
							count++;
						if (w2 == index1 && o2 == index2)
							count++;

					} else if (flag.equals("ll")) {
						if (o1 == index1 && o2 == index2)
							count++;

					} else if (flag.equals("lp")) {
						if (o1 == index1 && pred == index2)
							count++;
						if (o2 == index1 && pred == index2)
							count++;

					} else if (flag.equals("po")) {
						if (pred == index1
								&& example.observedRelation == index2)
							count++;
					}
					total += Math
							.exp((wordIndexLabelIndexMatrix[w1][o1]
									+ wordIndexLabelIndexMatrix[w2][o2]

									+ labelIndexLabelIndexMatrix[o1][o2]

									+ labelIndexPredictedNodeMatrix[o1][pred]
									+ labelIndexPredictedNodeMatrix[o2][pred] + predictedNodeObservedNodeMatrix[pred][example.observedRelation]))
							* count;

				}
			}
		}
		double bruteForceEquivalent = Math.log(total);
		System.out.println("derivative brute force result: "
				+ bruteForceEquivalent);
	}

	/**
	 * This function depends on computePartitionFunctionEquivalentExponent.
	 * 
	 * @param example
	 * @param flag
	 *            {"wl","ll","lp","po"}
	 * @param index1
	 * @param index2
	 * @return
	 */
	/**
	 * Changes to include observation as a variable
	 */
	private static double computePartitionFunctionDerivativeEquivalentExponent(
			TrainingExample example, String flag, int index1, int index2) {
		ArrayList<Integer> wordIndices = example.wordIndices;
		int chainLength = wordIndices.size();
		// basically, doesn't allow chain of one node
		if (chainLength <= 1) {
			System.err.println("chain length is not valid");
		}
		double[] prevExponents = new double[numOfAbstractLabels];
		double[] prevExponents2 = new double[2 * numOfAbstractLabels];
		// for the last else statement, optimization purpose
		double[][] prevExponents2_2d = new double[2 * numOfAbstractLabels][numOfAbstractLabels];
		double[][] arr2 = new double[numOfAbstractLabels][numOfAbstractLabels * 2];

		double[] succExponents = new double[numOfAbstractLabels];
		double[] temp = new double[numOfAbstractLabels];
		double[] chainEquivalentExponents = new double[numOfPredictedValues];
		// ****************************************************************************//
		double[] chainPredEquivalentExponents2 = new double[2
				* numOfPredictedValues * numOfObservedValues];
		// ****************************************************************************//

		int wordIndex;
		double totalEquivalentExponent;

		boolean wlFlag = flag.equals("wl");
		boolean llFlag = flag.equals("ll");
		boolean lpFlag = flag.equals("lp");

		double[] computedEquivalentExponentsArray;
		for (int pred = 0; pred < numOfPredictedValues; pred++) {
			for (int pos = 0; pos < chainLength; pos++) {
				wordIndex = example.wordIndices.get(pos);
				// for optimization purpose
				double[] wordIndexArray = wordIndexLabelIndexMatrix[wordIndex];

				if (pos == 0) {
					for (int succ = 0; succ < numOfAbstractLabels; succ++) {
						// succExponents[succ] = 0;
						for (int prev = 0; prev < numOfAbstractLabels; prev++) {
							// three edges here
							// if ((flag.equals("wl") && index1 == wordIndex &&
							// index2 == prev)
							// || (flag.equals("ll") && index1 == prev && index2
							// == succ)
							// || (flag.equals("lp") && index1 == prev && index2
							// == pred))
							if ((wlFlag && index1 == wordIndex && index2 == prev)
									|| (llFlag && index1 == prev && index2 == succ)
									|| (lpFlag && index1 == prev && index2 == pred))
								prevExponents[prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
										+ labelIndexPredictedNodeMatrix[prev][pred]
										+ labelIndexLabelIndexMatrix[prev][succ];
							else
								prevExponents[prev] = Double.NEGATIVE_INFINITY;
						}
						succExponents[succ] = computeEquivalentExponent(prevExponents);
					}

				} else if (pos == chainLength - 1) {
					for (int i = 0; i < numOfAbstractLabels; i++) {
						temp[i] = succExponents[i];
					}
					for (int prev = 0; prev < numOfAbstractLabels; prev++) {
						// the term using temp[i]
						prevExponents2[2 * prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
								+ labelIndexPredictedNodeMatrix[prev][pred]
								+ temp[prev];
						// the term using stored info
						// if ((flag.equals("wl") && index1 == wordIndex &&
						// index2 == prev)
						// || (flag.equals("lp") && index1 == prev && index2 ==
						// pred))
						if ((wlFlag && index1 == wordIndex && index2 == prev)
								|| (lpFlag && index1 == prev && index2 == pred))
							prevExponents2[2 * prev + 1] = wordIndexLabelIndexMatrix[wordIndex][prev]
									+ labelIndexPredictedNodeMatrix[prev][pred];
						else
							prevExponents2[2 * prev + 1] = Double.NEGATIVE_INFINITY;
						prevExponents2[2 * prev + 1] += computedEquivalentExponents[pred][pos - 1][prev];
					}
					chainEquivalentExponents[pred] = computeEquivalentExponent(prevExponents2);
				} else {
					// positions in the middle
					// "pos" & "pred" are parameters of outer loops

					for (int i = 0; i < numOfAbstractLabels; i++) {
						temp[i] = succExponents[i];
					}

					// for (int succ = 0; succ < numOfAbstractLabels; succ++) {
					// // succExponents[succ] = 0;
					// for (int prev = 0; prev < numOfAbstractLabels; prev++) {
					// // the term using temp[i]
					// prevExponents2[2 * prev] =
					// wordIndexLabelIndexMatrix[wordIndex][prev]
					// + labelIndexPredictedNodeMatrix[prev][pred]
					// + labelIndexLabelIndexMatrix[prev][succ]
					// + temp[prev];
					// // the term using stored info
					// // if ((flag.equals("wl") && index1 == wordIndex &&
					// // index2 == prev)
					// // || (flag.equals("ll") && index1 == prev && index2
					// // == succ)
					// // || (flag.equals("lp") && index1 == prev && index2
					// // == pred))
					// if ((wlFlag && index1 == wordIndex && index2 == prev)
					// || (llFlag && index1 == prev && index2 == succ)
					// || (lpFlag && index1 == prev && index2 == pred))
					// prevExponents2[2 * prev + 1] =
					// wordIndexLabelIndexMatrix[wordIndex][prev]
					// + labelIndexPredictedNodeMatrix[prev][pred]
					// + labelIndexLabelIndexMatrix[prev][succ];
					// else
					// prevExponents2[2 * prev + 1] = Double.NEGATIVE_INFINITY;
					// prevExponents2[2 * prev + 1] +=
					// computedEquivalentExponents[pred][pos - 1][prev];
					// }
					// succExponents[succ] =
					// computeEquivalentExponent(prevExponents2);
					// }

					computedEquivalentExponentsArray = computedEquivalentExponents[pred][pos - 1];

					for (int prev = 0; prev < numOfAbstractLabels; prev++) {
						for (int succ = 0; succ < numOfAbstractLabels; succ++) {
							// succExponents[succ] = 0;

							// the term using temp[i]
							prevExponents2_2d[2 * prev][succ] = wordIndexArray[prev]
									+ labelIndexPredictedNodeMatrix[prev][pred]
									+ labelIndexLabelIndexMatrix[prev][succ]
									+ temp[prev];
							if ((wlFlag && index1 == wordIndex && index2 == prev)
									|| (llFlag && index1 == prev && index2 == succ)
									|| (lpFlag && index1 == prev && index2 == pred))
								prevExponents2_2d[2 * prev + 1][succ] = wordIndexLabelIndexMatrix[wordIndex][prev]
										+ labelIndexPredictedNodeMatrix[prev][pred]
										+ labelIndexLabelIndexMatrix[prev][succ];
							else
								prevExponents2_2d[2 * prev + 1][succ] = Double.NEGATIVE_INFINITY;
							prevExponents2_2d[2 * prev + 1][succ] += computedEquivalentExponentsArray[prev];
						}
					}
					// succExponents[succ] =
					// computeEquivalentExponent(prevExponents2);

					for (int i = 0; i < 2 * numOfAbstractLabels; i++) {
						final double[] arr = prevExponents2_2d[i];
						for (int j = 0; j < numOfAbstractLabels; j++) {
							arr2[j][i] = arr[j];
						}
					}

					for (int i = 0; i < numOfAbstractLabels; i++) {
						succExponents[i] = computeEquivalentExponent(arr2[i]);
					}

				}
			}
			// ****************************************************************************//
			for (int ob = 0; ob < numOfObservedValues; ob++) {
				// term using info just computed
				chainPredEquivalentExponents2[pred * 4 + ob * 2] = chainEquivalentExponents[pred]
						+ predictedNodeObservedNodeMatrix[pred][ob];
				// term using stored info
				if (flag.equals("po") && index1 == pred
						&& index2 == example.observedRelation)
					chainPredEquivalentExponents2[pred * 4 + ob * 2 + 1] = predictedNodeObservedNodeMatrix[pred][ob];
				else
					chainPredEquivalentExponents2[pred * 4 + ob * 2 + 1] = Double.NEGATIVE_INFINITY;
				chainPredEquivalentExponents2[pred * 4 + ob * 2 + 1] += computedEquivalentExponentsWithPredictionAsParam[pred];
			}
			// ****************************************************************************//

			// for (int ob = 0; ob < numOfObservedValues; ob++) {
			// // term using info just computed
			// chainPredEquivalentExponents2[pred * 2] =
			// chainEquivalentExponents[pred]
			// +
			// predictedNodeObservedNodeMatrix[pred][example.observedRelation];
			// // term using stored info
			// if (flag.equals("po") && index1 == pred
			// && index2 == example.observedRelation)
			// chainPredEquivalentExponents2[pred * 2 + 1] =
			// predictedNodeObservedNodeMatrix[pred][example.observedRelation];
			// else
			// chainPredEquivalentExponents2[pred * 2 + 1] =
			// Double.NEGATIVE_INFINITY;
			// chainPredEquivalentExponents2[pred * 2 + 1] +=
			// computedEquivalentExponentsWithPredictionAsParam[pred];
			// }
		}
		totalEquivalentExponent = computeEquivalentExponent(chainPredEquivalentExponents2);
		return totalEquivalentExponent;
	}

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
		System.out.println("belief propagation result: "
				+ totalEquivalentExponent);
		return totalEquivalentExponent;
	}
}

// /**
// * Depends on computePartitionFunctionEquivalentExponent. For doing exact
// * denominator computation.
// *
// * @param example
// * @return
// */
// private static double getLogLikelihoodOfLabeledTrainingExample(
// TrainingExample example) {
// double ret = 0;
// example.chainLength = example.wordIndices.size();
// DenominatorExample edges;
// edges = getEdgesFromNodes(example);
// setTotalWeightsForDenomEx(edges);
// double numWeights = edges.totalWeights;
// // for test purpose
// System.out.println("numerator weights total is " + numWeights);
// double denomEquivalentExponent =
// computePartitionFunctionEquivalentExponent(example);
// // for test purpose
// System.out.println("denominator equivalent exponent is "
// + denomEquivalentExponent);
// ret = numWeights - denomEquivalentExponent;
// return ret;
// }
//
// private static void setTotalWeightsForDenomEx(DenominatorExample denomEx) {
// double totalWeights = 0;
// int prediction = denomEx.predictionObservationEdge.vertex1;
// int observation = denomEx.predictionObservationEdge.vertex2;
// totalWeights += predictedNodeObservedNodeMatrix[prediction][observation];
// Edge edge1, edge2;
// int vertex1, vertex2;
// for (int i = 0; i < denomEx.wordLabelEdges.size(); i++) {
// edge1 = denomEx.wordLabelEdges.get(i);
// vertex1 = edge1.vertex1;
// vertex2 = edge1.vertex2;
// totalWeights += wordIndexLabelIndexMatrix[vertex1][vertex2];
// edge2 = denomEx.labelPredictionEdges.get(i);
// vertex1 = edge2.vertex1;
// vertex2 = edge2.vertex2;
// totalWeights += labelIndexPredictedNodeMatrix[vertex1][vertex2];
// }
// for (int i = 0; i < denomEx.labelLabelEdges.size(); i++) {
// edge1 = denomEx.labelLabelEdges.get(i);
// vertex1 = edge1.vertex1;
// vertex2 = edge1.vertex2;
// totalWeights += labelIndexLabelIndexMatrix[vertex1][vertex2];
// }
// denomEx.totalWeights = totalWeights;
// // denomEx.totalWeightsExp = Math.exp(totalWeights);
// }
//
// private static DenominatorExample getEdgesFromNodes(
// TrainingExample trainingExample) {
// DenominatorExample ret = new DenominatorExample();
// int observation = trainingExample.observedRelation;
// int prediction = trainingExample.predictedRelation;
// int label, wordIndex, prevLabel = -1;
// ret.predictionObservationEdge = new Edge(prediction, observation);
// for (int i = 0; i < trainingExample.chainLength; i++) {
// label = trainingExample.labels.get(i);
// wordIndex = trainingExample.wordIndices.get(i);
// ret.wordLabelEdges.add(new Edge(wordIndex, label));
// ret.labelPredictionEdges.add(new Edge(label, prediction));
// if (i != 0)
// ret.labelLabelEdges.add(new Edge(prevLabel, label));
// prevLabel = label;
// }
// return ret;
// }

// /**
// * Uses belief propagation to calculate log of partition function.
// *
// * @param example
// * @return
// */
// /**
// * Changes to include observation as a variable.
// */
// private static double computePartitionFunctionEquivalentExponent(
// TrainingExample example) {
// ArrayList<Integer> wordIndices = example.wordIndices;
// int chainLength = wordIndices.size();
// if (chainLength == 1) {
// System.err
// .println("There is an error. No chain should be of length 1.");
// }
// int wordIndex;
//
// // added for computing derivatives
// // special case: last chain label with prediction node as function
// // parameter
// computedEquivalentExponents = new
// double[numOfPredictedValues][chainLength][numOfAbstractLabels];
// computedEquivalentExponentsWithPredictionAsParam = new
// double[numOfPredictedValues];
//
// double[] prevExponents = new double[numOfAbstractLabels];
// double[] succExponents = new double[numOfAbstractLabels];
// double[] temp = new double[numOfAbstractLabels];
// double[] chainEquivalentExponents = new double[numOfPredictedValues];
// //
// ****************************************************************************//
// double[] chainPredEquivalentExponents = new double[numOfPredictedValues
// * numOfObservedValues];
// //
// ****************************************************************************//
// double totalEquivalentExponent;
//
// for (int pred = 0; pred < numOfPredictedValues; pred++) {
// for (int pos = 0; pos < chainLength; pos++) {
// wordIndex = wordIndices.get(pos);
// // last label will not have label-label edge
// // first label position
// if (pos == 0) {
// for (int succ = 0; succ < numOfAbstractLabels; succ++) {
// // succExponents[succ] = 0;
// for (int prev = 0; prev < numOfAbstractLabels; prev++) {
// // prevExponents[prev] = 0;
// prevExponents[prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
// + labelIndexPredictedNodeMatrix[prev][pred]
// + labelIndexLabelIndexMatrix[prev][succ];
// }
// succExponents[succ] = computeEquivalentExponent(prevExponents);
// // added for computing derivative
// computedEquivalentExponents[pred][pos][succ] = succExponents[succ];
// }
// }
// // last label position
// else if (pos == chainLength - 1) {
// for (int i = 0; i < numOfAbstractLabels; i++) {
// temp[i] = succExponents[i];
// }
//
// for (int prev = 0; prev < numOfAbstractLabels; prev++) {
// // prevExponents[prev] = 0;
// prevExponents[prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
// + labelIndexPredictedNodeMatrix[prev][pred]
// + temp[prev];
// }
// chainEquivalentExponents[pred] = computeEquivalentExponent(prevExponents);
// // added for computing derivatives
// computedEquivalentExponentsWithPredictionAsParam[pred] =
// chainEquivalentExponents[pred];
// }
// // positions in between
// else {
// for (int i = 0; i < numOfAbstractLabels; i++) {
// temp[i] = succExponents[i];
// }
//
// for (int succ = 0; succ < numOfAbstractLabels; succ++) {
// // succExponents[succ] = 0;
// for (int prev = 0; prev < numOfAbstractLabels; prev++) {
// // prevExponents[prev] = 0;
// prevExponents[prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
// + labelIndexPredictedNodeMatrix[prev][pred]
// + labelIndexLabelIndexMatrix[prev][succ]
// + temp[prev];
// }
// succExponents[succ] = computeEquivalentExponent(prevExponents);
// // added for computing derivatives
// computedEquivalentExponents[pred][pos][succ] = succExponents[succ];
// }
// }
//
// }
// //
// ****************************************************************************//
// // !!! modification here to include observation as a variable
// // Also reduces complexity
// for (int ob = 0; ob < numOfObservedValues; ob++) {
// chainPredEquivalentExponents[pred * numOfObservedValues + ob] =
// chainEquivalentExponents[pred]
// + predictedNodeObservedNodeMatrix[pred][ob];
// }
// // chainPredEquivalentExponents[pred] =
// // chainEquivalentExponents[pred]
// // +
// // predictedNodeObservedNodeMatrix[pred][example.observedRelation];
// //
// ****************************************************************************//
//
// }
// totalEquivalentExponent =
// computeEquivalentExponent(chainPredEquivalentExponents);
// System.out.println("belief propagation result: "
// + totalEquivalentExponent);
// return totalEquivalentExponent;
// }

// /**
// * This function depends on computePartitionFunctionEquivalentExponent.
// *
// * @param example
// * @param flag
// * {"wl","ll","lp","po"}
// * @param index1
// * @param index2
// * @return
// */
// /**
// * Changes to include observation as a variable
// */
// private static double
// computePartitionFunctionDerivativeEquivalentExponent(
// TrainingExample example, String flag, int index1, int index2) {
// ArrayList<Integer> wordIndices = example.wordIndices;
// int chainLength = wordIndices.size();
// // basically, doesn't allow chain of one node
// if (chainLength <= 1) {
// System.err.println("chain length is not valid");
// }
// double[] prevExponents = new double[numOfAbstractLabels];
// double[] prevExponents2 = new double[2 * numOfAbstractLabels];
// double[] succExponents = new double[numOfAbstractLabels];
// double[] temp = new double[numOfAbstractLabels];
// double[] chainEquivalentExponents = new double[numOfPredictedValues];
// //
// ****************************************************************************//
// double[] chainPredEquivalentExponents2 = new double[2
// * numOfPredictedValues * numOfObservedValues];
// //
// ****************************************************************************//
//
// int wordIndex;
// double totalEquivalentExponent;
//
// for (int pred = 0; pred < numOfPredictedValues; pred++) {
// for (int pos = 0; pos < chainLength; pos++) {
// wordIndex = example.wordIndices.get(pos);
// if (pos == 0) {
// for (int succ = 0; succ < numOfAbstractLabels; succ++) {
// // succExponents[succ] = 0;
// for (int prev = 0; prev < numOfAbstractLabels; prev++) {
// // three edges here
// if ((flag.equals("wl") && index1 == wordIndex && index2 == prev)
// || (flag.equals("ll") && index1 == prev && index2 == succ)
// || (flag.equals("lp") && index1 == prev && index2 == pred))
// prevExponents[prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
// + labelIndexPredictedNodeMatrix[prev][pred]
// + labelIndexLabelIndexMatrix[prev][succ];
// else
// prevExponents[prev] = Double.NEGATIVE_INFINITY;
// }
// succExponents[succ] = computeEquivalentExponent(prevExponents);
// }
//
// } else if (pos == chainLength - 1) {
// for (int i = 0; i < numOfAbstractLabels; i++) {
// temp[i] = succExponents[i];
// }
// for (int prev = 0; prev < numOfAbstractLabels; prev++) {
// // the term using temp[i]
// prevExponents2[2 * prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
// + labelIndexPredictedNodeMatrix[prev][pred]
// + temp[prev];
// // the term using stored info
// if ((flag.equals("wl") && index1 == wordIndex && index2 == prev)
// || (flag.equals("lp") && index1 == prev && index2 == pred))
// prevExponents2[2 * prev + 1] = wordIndexLabelIndexMatrix[wordIndex][prev]
// + labelIndexPredictedNodeMatrix[prev][pred];
// else
// prevExponents2[2 * prev + 1] = Double.NEGATIVE_INFINITY;
// prevExponents2[2 * prev + 1] += computedEquivalentExponents[pred][pos -
// 1][prev];
// }
// chainEquivalentExponents[pred] =
// computeEquivalentExponent(prevExponents2);
// } else {
// // positions in the middle
// for (int i = 0; i < numOfAbstractLabels; i++) {
// temp[i] = succExponents[i];
// }
//
// for (int succ = 0; succ < numOfAbstractLabels; succ++) {
// // succExponents[succ] = 0;
// for (int prev = 0; prev < numOfAbstractLabels; prev++) {
// // the term using temp[i]
// prevExponents2[2 * prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
// + labelIndexPredictedNodeMatrix[prev][pred]
// + labelIndexLabelIndexMatrix[prev][succ]
// + temp[prev];
// // the term using stored info
// if ((flag.equals("wl") && index1 == wordIndex && index2 == prev)
// || (flag.equals("ll") && index1 == prev && index2 == succ)
// || (flag.equals("lp") && index1 == prev && index2 == pred))
// prevExponents2[2 * prev + 1] = wordIndexLabelIndexMatrix[wordIndex][prev]
// + labelIndexPredictedNodeMatrix[prev][pred]
// + labelIndexLabelIndexMatrix[prev][succ];
// else
// prevExponents2[2 * prev + 1] = Double.NEGATIVE_INFINITY;
// prevExponents2[2 * prev + 1] += computedEquivalentExponents[pred][pos -
// 1][prev];
// }
// succExponents[succ] = computeEquivalentExponent(prevExponents2);
// }
// }
// }
// //
// ****************************************************************************//
// for (int ob = 0; ob < numOfObservedValues; ob++) {
// // term using info just computed
// chainPredEquivalentExponents2[pred * 4 + ob * 2] =
// chainEquivalentExponents[pred]
// + predictedNodeObservedNodeMatrix[pred][ob];
// // term using stored info
// if (flag.equals("po") && index1 == pred
// && index2 == example.observedRelation)
// chainPredEquivalentExponents2[pred * 4 + ob * 2 + 1] =
// predictedNodeObservedNodeMatrix[pred][ob];
// else
// chainPredEquivalentExponents2[pred * 4 + ob * 2 + 1] =
// Double.NEGATIVE_INFINITY;
// chainPredEquivalentExponents2[pred * 4 + ob * 2 + 1] +=
// computedEquivalentExponentsWithPredictionAsParam[pred];
// }
// //
// ****************************************************************************//
//
// // for (int ob = 0; ob < numOfObservedValues; ob++) {
// // // term using info just computed
// // chainPredEquivalentExponents2[pred * 2] =
// // chainEquivalentExponents[pred]
// // +
// // predictedNodeObservedNodeMatrix[pred][example.observedRelation];
// // // term using stored info
// // if (flag.equals("po") && index1 == pred
// // && index2 == example.observedRelation)
// // chainPredEquivalentExponents2[pred * 2 + 1] =
// // predictedNodeObservedNodeMatrix[pred][example.observedRelation];
// // else
// // chainPredEquivalentExponents2[pred * 2 + 1] =
// // Double.NEGATIVE_INFINITY;
// // chainPredEquivalentExponents2[pred * 2 + 1] +=
// // computedEquivalentExponentsWithPredictionAsParam[pred];
// // }
// }
// totalEquivalentExponent =
// computeEquivalentExponent(chainPredEquivalentExponents2);
// return totalEquivalentExponent;
// }
