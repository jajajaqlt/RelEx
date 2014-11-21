package part2;

import java.io.BufferedReader; 
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;
import org.apache.commons.math3.util.CombinatoricsUtils;

public class GeneralRelationExtractor {

	/*
	 * weights
	 */
	public static double[][] wordIndexLabelIndexMatrix;
	public static double[][] labelIndexLabelIndexMatrix;
	public static double[][] labelIndexPredictedNodeMatrix;
	public static double[][] predictedNodeObservedNodeMatrix;

	/*
	 * counts
	 */
	public static double[][] aggregateWordIndexLabelIndexMatrix;
	public static double[][] aggregateLabelIndexLabelIndexMatrix;
	public static double[][] aggregateLabelIndexPredictedNodeMatrix;
	public static double[][] aggregatePredictedNodeObservedNodeMatrix;

	/*
	 * gradients
	 */
	public static double[][] gradientWordIndexLabelIndexMatrix;
	public static double[][] gradientLabelIndexLabelIndexMatrix;
	public static double[][] gradientLabelIndexPredictedNodeMatrix;
	public static double[][] gradientPredictedNodeObservedNodeMatrix;

	/*
	 * Sizes of labels
	 */
	public static int numOfAllCUIs;
	public static int numOfAbstractLabels = 6;
	public static int numOfPredictedValues = 2;
	public static int numOfObservedValues = 2;

	public static int numberOfKeyWords = 2;

	/*
	 * IO
	 */
	public static String inputFile = "synthetic9";
	public static String testFile = "synthetic9_test";
	public static boolean printCollections = false;
	public static boolean printWeightMatrices = false;
	public static boolean printWordLabelMatrices = true;
	public static boolean printWeightsRelatedStuff = false;
	public static int maximizationPrintoutInterval = 1;

	/*
	 * Testing
	 */
	public static int testingBurnInSteps = 50;

	/*
	 * Distribution
	 */
	public static EnumeratedIntegerDistribution labelDistribution;
	public static EnumeratedIntegerDistribution predictionDistribution;
	public static EnumeratedIntegerDistribution observationDistribution;

	/*
	 * Flags
	 */
	public static boolean isObservedVariableLatent = false;
	public static boolean isSyntheticData = true;
	// In practical situations, this flag should always be negative.
	public static boolean isLabelMatrixSymmetric = false;
	public static boolean isInitialWeightsUniform = true;
	// Used to label all words between relation1 and relation2 zero
	public static boolean labelWordsInBetweenZero = false;
	public static boolean initialPredictionObservationEquality = false;
	public static boolean preCalculateWeights = false;
	public static boolean isStratifiedDenominator = false;
	// Sets this flag to be true for normal case
	public static boolean isThereEdgeBetweenLabels = true;
	public static boolean isPredictionObservationWeightFixed = true;

	/*
	 * 
	 * Algorithm parameters
	 */
	public static int numOfExpectationSamples = 10;
	public static double stepSize = 0.0001;
	public static int initialBurnInSteps = 500;
	public static int samplingGap = 5;
	public static int emBurnInSteps = 50;
	public static int maxEStep = 40;
	public static int maxMStep = 100;
	public static double expectationTerminalPercentage = 0.01;
	public static double maximizationTerminalPercentage = 0.01;

	/*
	 * For doing exact denominator computation.
	 */

	// chainLength * possible labels
	// These two are initialized every time because of different chain lengths
	public static double[][][] computedEquivalentExponents;
	public static double[] computedEquivalentExponentsWithPredictionAsParam;

	/*
	 * Misc
	 */
	// how many relations are instances of treat relationship
	public static int treatRelation;
	/*
	 * 0: initial, 1: first increasing phase, 2: first decreasing phase, 3:
	 * second increasing phase, 4: closing phase, 5: ...
	 */
	public static int maxPhase;
	public static double predObEqualityInitVal = 3;
	public static double predObDiffInitVal = 1;

	public static void main(String[] args) throws Exception {

		ArrayList<TrainingExample> examples = new ArrayList<TrainingExample>();

		if (!isSyntheticData) {
			examples.addAll(makeTrainingExamplesFromRealData(inputFile));
		} else {
			// synthetic data example
			// 834 666 150 292(index1_content) 152 220 480 278 379 307 397 548
			// 546
			// 779 691 76 361 80
			// 141(index2_content) 71 343 904
			// 0(flag) 15(index1) 30(index2)

			examples.addAll(makeTrainingExamplesFromSyntheticData(inputFile));
			// after this, for synthetic data, the following is set
			// chainLength, observedRelation, relationWordIndex1,
			// relationWordIndex2, wordIndices
			// 'words' will not be set forever

		}
		// for calculating gradients
		assignDistinctWordIndicesField(examples);

		// Uniformly labels hidden variables
		assignInitialValuesForLatentVariables(examples, numOfAbstractLabels);
		// after this, predictedRelation and labels are set
		// denominatorExmaples will not be set until calculating derivatives

		/**
		 * Up to now, abstracts input into a set of model instances.
		 */

		// Initializes weight matrices
		initializeMatrices();

		//
		if (!isThereEdgeBetweenLabels)
			emptyLabelLabelMatrix();

		// Initial burns in
		System.out.println("initial burn-in starts. time(s): "
				+ System.currentTimeMillis() / 1000);
		for (int i = 0; i < initialBurnInSteps; i++) {
			System.out.println("Initial burn-in #" + i);
			burnIn(1, examples);
			// printExamples(examples);
			// pause();
		}
		System.out.println("initial burn-in terminates. time(s): "
				+ System.currentTimeMillis() / 1000);

		// Initial Printout
		if (printWeightsRelatedStuff) {
			// Prints weights
			printWeightMatrices();
			// Prints examples after burn-in
			printExamples(examples);
			countExampleEdges(examples);
			pause();
		}

		// Encapsulates examples into a TrainingExampleCollection objects
		TrainingExampleCollection exampleColletion = new TrainingExampleCollection();
		exampleColletion.examples = examples;
		ArrayList<TrainingExampleCollection> collections = new ArrayList<TrainingExampleCollection>();
		TrainingExampleCollection temp;

		/**
		 * End of initialization
		 */

		/**
		 * All likelihood terms below refer to Q function values.
		 */
		double prevExpectationLikelihood = 0, curExpectationLikelihood = 2, prevMaximizationLikelihood, curMaximizationLikelihood = 0;

		int EMIter = 0;

		while (true) {
			// sets to false when doing exact denominator computation
			if (preCalculateWeights) {
				assignAggregateMatrices(examples);
				// already sets total weights
				getStratifiedExamplesForExamples(examples);
				collections.add(exampleColletion);
				setPercentageForDenominatorExample(collections);
				preCalculateWeights = false;
			} else {
				if (EMIter == maxEStep) {
					System.out.println("reaching maximum expectation step");
					break;
				}
				System.out.println("This is start of EM iteration #" + EMIter
						+ ".");

				// Reinitialization
				// remove all samples for examples ('examples' is all training
				// examples we have)
				collections.clear();
				emptyAllAggregateMatrices();

				// Burn-in inside EM
				System.out.println("inside EM burn-in starts. time(s): "
						+ System.currentTimeMillis() / 1000);
				for (int i = 0; i < emBurnInSteps; i++) {
					System.out.println("inside EM burn-in #" + i);
					burnIn(1, examples);
					// printExamples(examples);
					// pause();
				}
				System.out.println("inside EM burn-in terminates. time(s): "
						+ System.currentTimeMillis() / 1000);

				// Sampling to approximate expectation (summation/integration)
				for (int i = 0; i < numOfExpectationSamples; i++) {
					temp = getTrainingExampleCollection(exampleColletion,
							samplingGap);
					collections.add(temp);
					assignAggregateMatrices(temp.examples);
					// already sets total weights
					if (isStratifiedDenominator)
						getStratifiedExamplesForExamples(temp.examples);
				}

				/**
				 * Great complexity here O(# of samples * sampling gap)
				 */
				// ??? adds or not
				// updateDenominatorsTotalWeights(collections);
				// See DenominatorExample.java
				if (isStratifiedDenominator)
					setPercentageForDenominatorExample(collections);
			}

			printWeightMatrices();
			printExamples(examples);
			countExampleEdges(examples);
			pause();

			// Checks EM termination condition and outputs current expectation
			// log likelihood figures
			prevExpectationLikelihood = curExpectationLikelihood;
			// curExpectationLikelihood = getLogLikelihood(collections);
			curExpectationLikelihood = getLogLikelihoodWithExactDenominator(collections);
			if (curExpectationLikelihood == 0) {
				System.out
						.println("Current expectation likelihood is 0. Break.");
				break;
			}
			System.out.println("previous expectation log likelihood is: "
					+ prevExpectationLikelihood);
			System.out.println("current expectation log likelihood is: "
					+ curExpectationLikelihood);
			System.out
					.println("expectation log likelihood percentage change is: "
							+ getChangePercentage(prevExpectationLikelihood,
									curExpectationLikelihood));
			pause();

			// Reinitialization
			curMaximizationLikelihood = curExpectationLikelihood;
			prevMaximizationLikelihood = 0;
			int MaxIter = 0;
			boolean breakFlag;
			maxPhase = 0;
			double change = 0;

			// Maximization
			while (true) {
				if (MaxIter == maxMStep) {
					System.out.println("reaching maximum maximization step: "
							+ MaxIter);
					break;
				}

				prevMaximizationLikelihood = curMaximizationLikelihood;

				// 1. calculate gradients
				// 2. update weight matrices
				// 3. recalculate weights for each denominator example
				// calculateGradients(collections);
				calculateGradientsWithExactDenominator(collections);
				updateWeightMatricesFromGradientMatrices();
				if (isStratifiedDenominator) {
					updateDenominatorsTotalWeights(collections);
					/**
					 * setPercentageForDenominatorExample relies on
					 * updateDenominatorsTotalWeights
					 **/
					setPercentageForDenominatorExample(collections);
				}

				// printWeightMatrices();

				curMaximizationLikelihood = getLogLikelihoodWithExactDenominator(collections);
				if (curMaximizationLikelihood == 0) {
					System.out
							.println("Current maximization likelihood is zero. Break.");
					break;
				}
				System.out.println("Maximization iteration #" + MaxIter);
				System.out.println("current maximization log likelihood is: "
						+ curMaximizationLikelihood);
				change = Math.abs(prevMaximizationLikelihood
						- curMaximizationLikelihood);
				System.out
						.println("maximization log likelihood percentage change is: "
								+ getChangePercentage(
										prevMaximizationLikelihood,
										curMaximizationLikelihood));
				pause();

				MaxIter++;
				// break point to test gradient matrices and get-total-weights
				// function
			}
			System.out.println("This is end of EM iteration #" + EMIter + ".");
			// if (printCollections)
			// printCollections(collections);
			// if (printWeightMatrices)
			// printWeightMatrices();
			EMIter++;
		}
		// final sampling
		temp = getTrainingExampleCollection(exampleColletion, samplingGap);
		// collections.clear();
		// collections.add(temp);
		// printCollections(collections);
		System.out.println("Final weight matrices.");
		printWeightMatrices();
		System.out.println("Final examples.");
		printExamples(examples);

		// synthetic data testing
		// **************************************************//
		ArrayList<TrainingExample> test_examples = new ArrayList<TrainingExample>();
		test_examples.addAll(makeTrainingExamplesFromSyntheticData(testFile));
		assignInitialValuesForLatentVariables(test_examples,
				numOfAbstractLabels);
		burnIn(testingBurnInSteps, test_examples);
		System.out.println("Testing result:");
		printExamples(test_examples);
		// **************************************************//

		// System.out.println("Final newly found relation.");
		// printNewlyFoundRelation(temp);

	}

	private static void emptyLabelLabelMatrix() {
		for (int i = 0; i < numOfAbstractLabels; i++) {
			for (int j = 0; j < numOfAbstractLabels; j++) {
				labelIndexLabelIndexMatrix[i][j] = 0;
			}
		}
	}

	private static void initializeMatrices() {
		// false values here are constants
		wordIndexLabelIndexMatrix = makeMatrixWithInitialValues(false,
				numOfAllCUIs, numOfAbstractLabels, isInitialWeightsUniform);
		labelIndexLabelIndexMatrix = makeMatrixWithInitialValues(
				isLabelMatrixSymmetric, numOfAbstractLabels,
				numOfAbstractLabels, isInitialWeightsUniform);
		labelIndexPredictedNodeMatrix = makeMatrixWithInitialValues(false,
				numOfAbstractLabels, numOfPredictedValues,
				isInitialWeightsUniform);
		predictedNodeObservedNodeMatrix = makeMatrixWithInitialValues(false,
				numOfPredictedValues, numOfObservedValues,
				isInitialWeightsUniform);

		predictedNodeObservedNodeMatrix[0][0] = predObEqualityInitVal;
		predictedNodeObservedNodeMatrix[0][1] = predObDiffInitVal;
		predictedNodeObservedNodeMatrix[1][0] = predObDiffInitVal;
		predictedNodeObservedNodeMatrix[1][1] = predObEqualityInitVal;

		// Initializes count matrices
		aggregateWordIndexLabelIndexMatrix = new double[numOfAllCUIs][numOfAbstractLabels];
		aggregateLabelIndexLabelIndexMatrix = new double[numOfAbstractLabels][numOfAbstractLabels];
		aggregateLabelIndexPredictedNodeMatrix = new double[numOfAbstractLabels][numOfPredictedValues];
		aggregatePredictedNodeObservedNodeMatrix = new double[numOfPredictedValues][numOfObservedValues];

		// Initializes gradient matrices
		gradientWordIndexLabelIndexMatrix = new double[numOfAllCUIs][numOfAbstractLabels];
		gradientLabelIndexLabelIndexMatrix = new double[numOfAbstractLabels][numOfAbstractLabels];
		gradientLabelIndexPredictedNodeMatrix = new double[numOfAbstractLabels][numOfPredictedValues];
		gradientPredictedNodeObservedNodeMatrix = new double[numOfPredictedValues][numOfObservedValues];
	}

	private static void assignDistinctWordIndicesField(
			ArrayList<TrainingExample> examples) {
		// TODO Auto-generated method stub
		TrainingExample example;
		ArrayList<Integer> distinctWordIndices;
		int wordIndex;
		for (int i = 0; i < examples.size(); i++) {
			example = examples.get(i);
			distinctWordIndices = new ArrayList<Integer>();
			for (int j = 0; j < example.wordIndices.size(); j++) {
				wordIndex = example.wordIndices.get(j);
				if (!distinctWordIndices.contains(wordIndex))
					distinctWordIndices.add(wordIndex);
			}
			example.distinctWordIndices = distinctWordIndices;
		}
	}

	/**
	 * When doing exact denominator computation
	 * 
	 * @param collections
	 * @return
	 */
	private static double getLogLikelihoodWithExactDenominator(
			ArrayList<TrainingExampleCollection> collections) {
		TrainingExampleCollection collection;
		TrainingExample example;
		double totalLogLikelihood = 0;
		for (int i = 0; i < collections.size(); i++) {
			collection = collections.get(i);
			for (int j = 0; j < collection.examples.size(); j++) {
				example = collection.examples.get(j);
				totalLogLikelihood += getLogLikelihoodOfLabeledTrainingExample(example);
			}
		}
		return totalLogLikelihood;
	}

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
	// double[] chainEquivalentExponents = new double[numOfObservedValues];
	// double[] chainPredEquivalentExponents2 = new double[2 *
	// numOfObservedValues];
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
	// // term using info just computed
	// chainPredEquivalentExponents2[pred * 2] = chainEquivalentExponents[pred]
	// + predictedNodeObservedNodeMatrix[pred][example.observedRelation];
	// // term using stored info
	// if (flag.equals("po") && index1 == pred
	// && index2 == example.observedRelation)
	// chainPredEquivalentExponents2[pred * 2 + 1] =
	// predictedNodeObservedNodeMatrix[pred][example.observedRelation];
	// else
	// chainPredEquivalentExponents2[pred * 2 + 1] = Double.NEGATIVE_INFINITY;
	// chainPredEquivalentExponents2[pred * 2 + 1] +=
	// computedEquivalentExponentsWithPredictionAsParam[pred];
	// }
	// totalEquivalentExponent =
	// computeEquivalentExponent(chainPredEquivalentExponents2);
	// return totalEquivalentExponent;
	// }
	//
	// /**
	// * Uses belief propagation to calculate log of partition function.
	// *
	// * @param example
	// * @return
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
	// double[] chainEquivalentExponents = new double[numOfObservedValues];
	// double[] chainPredEquivalentExponents = new double[numOfObservedValues];
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
	// chainEquivalentExponents[pred] =
	// computeEquivalentExponent(prevExponents);
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
	// // !!! Sum_o Sum_p exp(po) Sum_L
	// chainPredEquivalentExponents[pred] = chainEquivalentExponents[pred]
	// + predictedNodeObservedNodeMatrix[pred][example.observedRelation];
	// }
	// totalEquivalentExponent =
	// computeEquivalentExponent(chainPredEquivalentExponents);
	// // System.out.println("belief propagation result: "
	// // + totalEquivalentExponent);
	// return totalEquivalentExponent;
	// }

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
		double[] succExponents = new double[numOfAbstractLabels];
		double[] temp = new double[numOfAbstractLabels];
		double[] chainEquivalentExponents = new double[numOfPredictedValues];
		// ****************************************************************************//
		double[] chainPredEquivalentExponents2 = new double[2
				* numOfPredictedValues * numOfObservedValues];
		// ****************************************************************************//

		int wordIndex;
		double totalEquivalentExponent;

		for (int pred = 0; pred < numOfPredictedValues; pred++) {
			for (int pos = 0; pos < chainLength; pos++) {
				wordIndex = example.wordIndices.get(pos);
				if (pos == 0) {
					for (int succ = 0; succ < numOfAbstractLabels; succ++) {
						// succExponents[succ] = 0;
						for (int prev = 0; prev < numOfAbstractLabels; prev++) {
							// three edges here
							if ((flag.equals("wl") && index1 == wordIndex && index2 == prev)
									|| (flag.equals("ll") && index1 == prev && index2 == succ)
									|| (flag.equals("lp") && index1 == prev && index2 == pred))
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
						if ((flag.equals("wl") && index1 == wordIndex && index2 == prev)
								|| (flag.equals("lp") && index1 == prev && index2 == pred))
							prevExponents2[2 * prev + 1] = wordIndexLabelIndexMatrix[wordIndex][prev]
									+ labelIndexPredictedNodeMatrix[prev][pred];
						else
							prevExponents2[2 * prev + 1] = Double.NEGATIVE_INFINITY;
						prevExponents2[2 * prev + 1] += computedEquivalentExponents[pred][pos - 1][prev];
					}
					chainEquivalentExponents[pred] = computeEquivalentExponent(prevExponents2);
				} else {
					// positions in the middle
					for (int i = 0; i < numOfAbstractLabels; i++) {
						temp[i] = succExponents[i];
					}

					for (int succ = 0; succ < numOfAbstractLabels; succ++) {
						// succExponents[succ] = 0;
						for (int prev = 0; prev < numOfAbstractLabels; prev++) {
							// the term using temp[i]
							prevExponents2[2 * prev] = wordIndexLabelIndexMatrix[wordIndex][prev]
									+ labelIndexPredictedNodeMatrix[prev][pred]
									+ labelIndexLabelIndexMatrix[prev][succ]
									+ temp[prev];
							// the term using stored info
							if ((flag.equals("wl") && index1 == wordIndex && index2 == prev)
									|| (flag.equals("ll") && index1 == prev && index2 == succ)
									|| (flag.equals("lp") && index1 == prev && index2 == pred))
								prevExponents2[2 * prev + 1] = wordIndexLabelIndexMatrix[wordIndex][prev]
										+ labelIndexPredictedNodeMatrix[prev][pred]
										+ labelIndexLabelIndexMatrix[prev][succ];
							else
								prevExponents2[2 * prev + 1] = Double.NEGATIVE_INFINITY;
							prevExponents2[2 * prev + 1] += computedEquivalentExponents[pred][pos - 1][prev];
						}
						succExponents[succ] = computeEquivalentExponent(prevExponents2);
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
		// System.out.println("belief propagation result: "
		// + totalEquivalentExponent);
		return totalEquivalentExponent;
	}

	/**
	 * Depends on computePartitionFunctionEquivalentExponent. For doing exact
	 * denominator computation.
	 * 
	 * @param example
	 * @return
	 */
	private static double getLogLikelihoodOfLabeledTrainingExample(
			TrainingExample example) {
		double ret = 0;
		example.chainLength = example.wordIndices.size();
		DenominatorExample edges;
		edges = getEdgesFromNodes(example);
		setTotalWeightsForDenomEx(edges);
		double numWeights = edges.totalWeights;
		// for test purpose
		// System.out.println("numerator weights total is " + numWeights);
		double denomEquivalentExponent = computePartitionFunctionEquivalentExponent(example);
		// for test purpose
		// System.out.println("denominator equivalent exponent is "
		// + denomEquivalentExponent);
		ret = numWeights - denomEquivalentExponent;
		return ret;
	}

	/**
	 * Makes use of code related to aggregate matrices Copies from
	 * assginAggregateMatrices
	 * 
	 * @param examples
	 */
	private static void countExampleEdges(ArrayList<TrainingExample> examples) {
		TrainingExample example;
		ArrayList<Integer> wordIndices;
		ArrayList<Integer> labels;
		int observedRelation;
		int predictedRelation;
		int wordIndex;
		int label;
		int prevLabel = -1;

		double[][] aggregateWordIndexLabelIndexMatrix = new double[numOfAllCUIs][numOfAbstractLabels];
		double[][] aggregateLabelIndexLabelIndexMatrix = new double[numOfAbstractLabels][numOfAbstractLabels];
		double[][] aggregateLabelIndexPredictedNodeMatrix = new double[numOfAbstractLabels][numOfPredictedValues];
		double[][] aggregatePredictedNodeObservedNodeMatrix = new double[numOfPredictedValues][numOfObservedValues];

		for (int i = 0; i < examples.size(); i++) {
			example = examples.get(i);
			wordIndices = example.wordIndices;
			labels = example.labels;
			observedRelation = example.observedRelation;
			predictedRelation = example.predictedRelation;
			aggregatePredictedNodeObservedNodeMatrix[predictedRelation][observedRelation] += 1;
			for (int j = 0; j < wordIndices.size(); j++) {
				wordIndex = wordIndices.get(j);
				label = labels.get(j);
				aggregateWordIndexLabelIndexMatrix[wordIndex][label] += 1;
				aggregateLabelIndexPredictedNodeMatrix[label][predictedRelation] += 1;
				if (j != 0) {
					aggregateLabelIndexLabelIndexMatrix[prevLabel][label] += 1;
					// ??? label label symmetry
					if (prevLabel != label && isLabelMatrixSymmetric)
						aggregateLabelIndexLabelIndexMatrix[label][prevLabel] += 1;
				}
				prevLabel = label;
			}
		}

		System.out.println();
		System.out.println("word-label count matrix:");
		for (int i = 0; i < numOfAllCUIs; i++) {
			System.out.print("[");
			for (int j = 0; j < numOfAbstractLabels; j++) {
				System.out.print("(" + aggregateWordIndexLabelIndexMatrix[i][j]
						+ ") ");
			}
			System.out.print("] ");
		}
		System.out.println();
		System.out.println("label-label count matrix");
		for (int i = 0; i < numOfAbstractLabels; i++) {
			System.out.print("[");
			for (int j = 0; j < numOfAbstractLabels; j++) {
				System.out.print("("
						+ aggregateLabelIndexLabelIndexMatrix[i][j] + ") ");
			}
			System.out.print("] ");
		}
		System.out.println();
		System.out.println("label-prediction count matrix:");
		for (int i = 0; i < numOfAbstractLabels; i++) {
			System.out.print("[");
			for (int j = 0; j < numOfPredictedValues; j++) {
				System.out.print("("
						+ aggregateLabelIndexPredictedNodeMatrix[i][j] + ") ");
			}
			System.out.print("] ");
		}
		System.out.println();
		System.out.println("prediction-observation count matrix:");
		for (int i = 0; i < numOfPredictedValues; i++) {
			System.out.print("[");
			for (int j = 0; j < numOfObservedValues; j++) {
				System.out
						.print("("
								+ aggregatePredictedNodeObservedNodeMatrix[i][j]
								+ ") ");
			}
			System.out.print("] ");
		}
		System.out.println();
	}

	private static void printNewlyFoundRelation(TrainingExampleCollection temp) {
		// TrainingExampleCollection collection;
		// ArrayList<DenominatorExample> denominators;
		DenominatorExample denominator;
		TrainingExample example;
		String twoTabs = "\t", threeTabs = "\t\t", indentation;
		ArrayList<Integer> indices;
		Edge edge;
		// String out;
		for (int j = 0; j < temp.examples.size(); j++) {
			// out = "";
			indentation = twoTabs;
			example = temp.examples.get(j);
			if (example.observedRelation == 0 && example.predictedRelation == 1) {
				System.out.println(indentation + "example#" + j);
				System.out.println(indentation + example.observedRelation);
				System.out.println(indentation + example.predictedRelation);

				indices = example.labels;
				System.out.print(indentation);
				for (int j2 = 0; j2 < example.chainLength; j2++) {
					System.out.print(indices.get(j2) + " ");
				}
				System.out.println();

				indices = example.wordIndices;
				System.out.print(indentation);
				for (int j2 = 0; j2 < example.chainLength; j2++) {
					System.out.print(indices.get(j2) + " ");
				}
				System.out.println();

				for (int k = 0; k < example.denominatorExamples.size(); k++) {
					indentation = threeTabs;
					denominator = example.denominatorExamples.get(k);
					System.out.println(indentation + "denominator#" + k);
					System.out.println(indentation + "percentage: "
							+ denominator.percentage);
					System.out.println(indentation + "total weights: "
							+ denominator.totalWeights);
					System.out.println(indentation
							+ denominator.predictionObservationEdge.vertex2);
					System.out.println(indentation
							+ denominator.predictionObservationEdge.vertex1);
					System.out.print(indentation);
					for (int k2 = 0; k2 < denominator.labelLabelEdges.size(); k2++) {
						edge = denominator.labelLabelEdges.get(k2);
						if (k2 == 0) {
							System.out.print(edge.vertex1 + " ");
						}
						System.out.print(edge.vertex2 + " ");
					}
					System.out.println();
					System.out.print(indentation);
					for (int k2 = 0; k2 < denominator.wordLabelEdges.size(); k2++) {
						edge = denominator.wordLabelEdges.get(k2);
						System.out.print(edge.vertex1 + " ");
					}
					System.out.println();
				}
			}
		}
		System.out.println();
	}

	private static boolean setMaxPhase(double prevMaximizationLikelihood,
			double curMaximizationLikelihood) {
		boolean ret = false;
		if (maxPhase == 0
				&& curMaximizationLikelihood > prevMaximizationLikelihood)
			maxPhase = 1;
		if (maxPhase == 1
				&& curMaximizationLikelihood < prevMaximizationLikelihood)
			maxPhase = 2;
		if (maxPhase == 2
				&& curMaximizationLikelihood > prevMaximizationLikelihood)
			maxPhase = 3;
		if (maxPhase == 3
				&& curMaximizationLikelihood < prevMaximizationLikelihood) {
			maxPhase = 4;
			ret = true;
		}
		return ret;
	}

	/**
	 * Sets field variables for TrainingExample objects. Sets numOfAllCUIs field
	 * variable.
	 * 
	 * @param fName
	 * @return
	 * @throws Exception
	 */
	private static ArrayList<TrainingExample> makeTrainingExamplesFromSyntheticData(
			String fName) throws Exception {
		ArrayList<TrainingExample> ret = new ArrayList<TrainingExample>();
		GeneratingSyntheticData p = new GeneratingSyntheticData();
		numOfAllCUIs = p.countDictSize(inputFile);
		BufferedReader br = new BufferedReader(new FileReader(fName));
		String line;
		String[] strs;
		int index;
		TrainingExample example;
		try {
			while ((line = br.readLine()) != null) {
				example = new TrainingExample();
				strs = line.split("\\s");
				for (int i = 0; i < strs.length; i++) {
					index = Integer.parseInt(strs[i]);
					// example.wordIndices.set(i, index);
					example.wordIndices.add(index);
				}
				example.chainLength = strs.length;
				line = br.readLine();
				strs = line.split("\\s");
				example.observedRelation = Integer.parseInt(strs[0]);
				example.relationWordIndex1 = Integer.parseInt(strs[1]);
				example.relationWordIndex2 = Integer.parseInt(strs[2]);
				ret.add(example);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}

		br.close();
		return ret;
	}

	private static ArrayList<TrainingExample> makeTrainingExamplesFromRealData(
			String fName) throws Exception {
		ArrayList<TrainingExample> examples = new ArrayList<TrainingExample>();
		String line, firstLine;
		boolean isCUILine = true;
		String[] CUIs = null, strIndices, allCUIs;
		int[] numIndices, CUIIndices = null;
		Map<String, Integer> CUIIndexMap = new HashMap<String, Integer>();
		Map<Integer, String> indexCUIMap = new HashMap<Integer, String>();

		// body
		BufferedReader br = new BufferedReader(new FileReader(inputFile));
		firstLine = br.readLine();
		numOfAllCUIs = Integer.parseInt(firstLine);
		allCUIs = br.readLine().split("\\s");
		for (int i = 0; i < numOfAllCUIs; i++) {
			CUIIndexMap.put(allCUIs[i], i);
			indexCUIMap.put(i, allCUIs[i]);
		}

		while ((line = br.readLine()) != null) {

			if (isCUILine) {
				CUIs = line.split("\\s");
				CUIIndices = new int[CUIs.length];
				for (int i = 0; i < CUIIndices.length; i++) {
					CUIIndices[i] = CUIIndexMap.get(CUIs[i]);
				}
				isCUILine = false;
			} else {
				if (!line.equals("")) {
					strIndices = line.split("\\s");
					numIndices = new int[strIndices.length];
					for (int i = 0; i < strIndices.length; i++) {
						numIndices[i] = Integer.parseInt(strIndices[i]);
					}
					examples.addAll(makeTrainingExamples(CUIs, CUIIndices,
							numIndices));
				} else
					examples.addAll(makeTrainingExamples(CUIs, CUIIndices, null));
				isCUILine = true;

			}
		}
		br.close();
		if (!isObservedVariableLatent) {
			Map<String, ArrayList<String>> relationMap = getRelationMatchings();
			labelObservedRelationNode(examples, relationMap);
			System.out.println("total count of treat relations is: "
					+ treatRelation);
		}
		return examples;
	}

	private static void printExamples(ArrayList<TrainingExample> examples) {
		TrainingExample example;
		ArrayList<Integer> indices;
		for (int j = 0; j < examples.size(); j++) {
			example = examples.get(j);
			System.out.println("example#" + j);
			System.out.println(example.observedRelation);
			System.out.println(example.predictedRelation);

			indices = example.labels;
			for (int j2 = 0; j2 < example.chainLength; j2++) {
				System.out.print(indices.get(j2) + " ");
			}
			System.out.println();

			indices = example.wordIndices;
			for (int j2 = 0; j2 < example.chainLength; j2++) {
				System.out.print(indices.get(j2) + " ");
			}
			System.out.println();
		}
		System.out.println();
	}

	private static void printCollections(
			ArrayList<TrainingExampleCollection> collections) {
		TrainingExampleCollection collection;
		ArrayList<DenominatorExample> denominators;
		DenominatorExample denominator;
		TrainingExample example;
		String tab = "\t", twoTabs = "\t\t", threeTabs = "\t\t\t", indentation;
		ArrayList<Integer> indices;
		Edge edge;
		System.out.println("Collections");
		for (int i = 0; i < collections.size(); i++) {
			indentation = tab;
			collection = collections.get(i);
			System.out.println(indentation + "collection#" + i);
			for (int j = 0; j < collection.examples.size(); j++) {
				indentation = twoTabs;
				example = collection.examples.get(j);
				System.out.println(indentation + "example#" + j);
				System.out.println(indentation + example.observedRelation);
				System.out.println(indentation + example.predictedRelation);

				indices = example.labels;
				System.out.print(indentation);
				for (int j2 = 0; j2 < example.chainLength; j2++) {
					System.out.print(indices.get(j2) + " ");
				}
				System.out.println();

				indices = example.wordIndices;
				System.out.print(indentation);
				for (int j2 = 0; j2 < example.chainLength; j2++) {
					System.out.print(indices.get(j2) + " ");
				}
				System.out.println();

				for (int k = 0; k < example.denominatorExamples.size(); k++) {
					indentation = threeTabs;
					denominator = example.denominatorExamples.get(k);
					System.out.println(indentation + "denominator#" + k);
					System.out.println(indentation + "percentage: "
							+ denominator.percentage);
					System.out.println(indentation + "total weights: "
							+ denominator.totalWeights);
					System.out.println(indentation
							+ denominator.predictionObservationEdge.vertex2);
					System.out.println(indentation
							+ denominator.predictionObservationEdge.vertex1);
					System.out.print(indentation);
					for (int k2 = 0; k2 < denominator.labelLabelEdges.size(); k2++) {
						edge = denominator.labelLabelEdges.get(k2);
						if (k2 == 0) {
							System.out.print(edge.vertex1 + " ");
						}
						System.out.print(edge.vertex2 + " ");
					}
					System.out.println();
					System.out.print(indentation);
					for (int k2 = 0; k2 < denominator.wordLabelEdges.size(); k2++) {
						edge = denominator.wordLabelEdges.get(k2);
						System.out.print(edge.vertex1 + " ");
					}
					System.out.println();
				}

			}
		}
	}

	private static void printWeightMatrices() {
		System.out.println("word-label aggregate array: ");
		// keywords[i]: sum of weights of all key words to label i
		double[] keyWords = new double[numOfAbstractLabels];
		// keywords[i]: sum of weights of all other words (except key words) to
		// label i
		double[] otherWords = new double[numOfAbstractLabels];

		for (int i = 0; i < numberOfKeyWords; i++) {
			for (int j = 0; j < numOfAbstractLabels; j++) {
				keyWords[j] += wordIndexLabelIndexMatrix[i][j];
			}
		}

		for (int i = numberOfKeyWords; i < numOfAllCUIs; i++) {
			for (int j = 0; j < numOfAbstractLabels; j++) {
				otherWords[j] += wordIndexLabelIndexMatrix[i][j];
			}
		}

		for (int i = 0; i < otherWords.length; i++) {
			keyWords[i] /= numberOfKeyWords;
			otherWords[i] /= (numOfAllCUIs - numberOfKeyWords);
		}

		System.out.println("Keywords to labels: ");
		for (int i = 0; i < keyWords.length; i++) {
			System.out.print("" + keyWords[i] + " ");
		}

		System.out.println();

		System.out.println("Other words to labels: ");
		for (int i = 0; i < otherWords.length; i++) {
			System.out.print("" + otherWords[i] + " ");
		}

		System.out.println();

		if (printWordLabelMatrices) {
			System.out.println("word-label matrix:");
			for (int i = 0; i < numOfAllCUIs; i++) {
				System.out.print("[");
				for (int j = 0; j < numOfAbstractLabels; j++) {
					System.out.print("(" + wordIndexLabelIndexMatrix[i][j]
							+ ") ");
				}
				System.out.print("] ");
			}
			System.out.println();
		}

		System.out.println("label-label matrix:");
		for (int i = 0; i < numOfAbstractLabels; i++) {
			System.out.print("[");
			for (int j = 0; j < numOfAbstractLabels; j++) {
				System.out.print("(" + labelIndexLabelIndexMatrix[i][j] + ") ");
			}
			System.out.print("] ");
		}
		System.out.println();
		System.out.println("label-prediction matrix:");
		for (int i = 0; i < numOfAbstractLabels; i++) {
			System.out.print("[");
			for (int j = 0; j < numOfPredictedValues; j++) {
				System.out.print("(" + labelIndexPredictedNodeMatrix[i][j]
						+ ") ");
			}
			System.out.print("] ");
		}
		System.out.println();
		System.out.println("prediction-observation matrix:");
		for (int i = 0; i < numOfPredictedValues; i++) {
			System.out.print("[");
			for (int j = 0; j < numOfObservedValues; j++) {
				System.out.print("(" + predictedNodeObservedNodeMatrix[i][j]
						+ ") ");
			}
			System.out.print("] ");
		}
		System.out.println();
	}

	private static void updateDenominatorsTotalWeights(
			ArrayList<TrainingExampleCollection> collections) {
		TrainingExampleCollection collection;
		ArrayList<DenominatorExample> denominators;
		DenominatorExample denominator;
		for (int i = 0; i < collections.size(); i++) {
			collection = collections.get(i);
			for (int j = 0; j < collection.examples.size(); j++) {
				denominators = collection.examples.get(j).denominatorExamples;
				for (int k = 0; k < denominators.size(); k++) {
					denominator = denominators.get(k);
					setTotalWeightsForDenomEx(denominator);
				}
			}
		}
	}

	private static void updateWeightMatricesFromGradientMatrices() {
		for (int i = 0; i < numOfAllCUIs; i++) {
			for (int j = 0; j < numOfAbstractLabels; j++) {
				wordIndexLabelIndexMatrix[i][j] += stepSize
						* gradientWordIndexLabelIndexMatrix[i][j];
			}
		}
		for (int i = 0; i < numOfAbstractLabels; i++) {
			for (int j = 0; j < numOfPredictedValues; j++) {
				labelIndexPredictedNodeMatrix[i][j] += stepSize
						* gradientLabelIndexPredictedNodeMatrix[i][j];
			}
		}
		for (int i = 0; i < numOfAbstractLabels; i++) {
			for (int j = 0; j < numOfAbstractLabels; j++) {
				labelIndexLabelIndexMatrix[i][j] += stepSize
						* gradientLabelIndexLabelIndexMatrix[i][j];
			}
		}
		for (int i = 0; i < numOfPredictedValues; i++) {
			for (int j = 0; j < numOfObservedValues; j++) {
				predictedNodeObservedNodeMatrix[i][j] += stepSize
						* gradientPredictedNodeObservedNodeMatrix[i][j];
			}
		}
	}

	private static void calculateGradientsWithExactDenominator(
			ArrayList<TrainingExampleCollection> collections) {
		// for each EM, empty first, then sample, then assign, finally copy.
		// counts of all samples of all examples
		copyValuesFromAggregateMatricesToGradientMatrices();
		TrainingExampleCollection collection;
		TrainingExample example;
		int wordIndex;
		double partitionFunctionExponent, partitionFunctionDerivativeExponent;
		int assignedObservation;

		int numOfCollections = collections.size();
		System.out.println("Calculating gradients, starting time is: "
				+ System.currentTimeMillis() / 1000 + "s");
		// for every sample of example, the derivative of z(x,p) is same. So, *
		// numOfC
		collection = collections.get(0);

		// for (int i = 0; i < collections.size(); i++) {
		// System.out.println("Calculating gradients, collection #" + i
		// + ", starting time is: " + System.currentTimeMillis()
		// / 1000 + "s");
		// collection = collections.get(i);
		for (int j = 0; j < collection.examples.size(); j++) {
			// System.out.println("Calculating gradients, example #" + j);
			example = collection.examples.get(j);
			// Only calculates partition function for each example
			partitionFunctionExponent = computePartitionFunctionEquivalentExponent(example);
			// word-label edges
			// |distinct words in word chain| * |labels|
			for (int k = 0; k < example.distinctWordIndices.size(); k++) {
				wordIndex = example.distinctWordIndices.get(k);
				for (int l = 0; l < numOfAbstractLabels; l++) {
					partitionFunctionDerivativeExponent = computePartitionFunctionDerivativeEquivalentExponent(
							example, "wl", wordIndex, l);
					gradientWordIndexLabelIndexMatrix[wordIndex][l] -= Math
							.exp(partitionFunctionDerivativeExponent
									- partitionFunctionExponent)
							* numOfCollections;
				}
			}

			// label-label edges
			// if (isThereEdgeBetweenLabels) {
			// for (int l1 = 0; l1 < numOfAbstractLabels; l1++) {
			// for (int l2 = 0; l2 < numOfAbstractLabels; l2++) {
			// partitionFunctionDerivativeExponent =
			// computePartitionFunctionDerivativeEquivalentExponent(
			// example, "ll", l1, l2);
			// gradientLabelIndexLabelIndexMatrix[l1][l2] -= Math
			// .exp(partitionFunctionDerivativeExponent
			// - partitionFunctionExponent)
			// * numOfCollections;
			// }
			// }
			// }

			if (isThereEdgeBetweenLabels) {
				for (int l1 = 0; l1 < numOfAbstractLabels; l1++) {
					for (int l2 = 0; l2 < numOfAbstractLabels; l2++) {
						partitionFunctionDerivativeExponent = computePartitionFunctionDerivativeEquivalentExponent(
								example, "ll", l1, l2);
						gradientLabelIndexLabelIndexMatrix[l1][l2] -= Math
								.exp(partitionFunctionDerivativeExponent
										- partitionFunctionExponent)
								* numOfCollections;
					}
				}
			}
			// label-prediction edges]
			for (int l = 0; l < numOfAbstractLabels; l++) {
				for (int pred = 0; pred < numOfPredictedValues; pred++) {
					partitionFunctionDerivativeExponent = computePartitionFunctionDerivativeEquivalentExponent(
							example, "lp", l, pred);
					gradientLabelIndexPredictedNodeMatrix[l][pred] -= Math
							.exp(partitionFunctionDerivativeExponent
									- partitionFunctionExponent)
							* numOfCollections;
				}
			}

			// prediction-observation edge
			// |predictions| * |assigned observation value| <- 1
			if (!isPredictionObservationWeightFixed) {
				assignedObservation = example.observedRelation;
				for (int pred = 0; pred < numOfPredictedValues; pred++) {
					partitionFunctionDerivativeExponent = computePartitionFunctionDerivativeEquivalentExponent(
							example, "po", pred, assignedObservation);
					gradientPredictedNodeObservedNodeMatrix[pred][assignedObservation] -= Math
							.exp(partitionFunctionDerivativeExponent
									- partitionFunctionExponent)
							* numOfCollections;
				}
			}

		}
		System.out.println("Calculating gradients, ending time is: "
				+ System.currentTimeMillis() / 1000 + "s");
		// }

	}

	private static void calculateGradients(
			ArrayList<TrainingExampleCollection> collections) {
		copyValuesFromAggregateMatricesToGradientMatrices();
		TrainingExampleCollection collection;
		TrainingExample example;
		DenominatorExample denominator;
		ArrayList<Edge> wordLabelEdges;
		ArrayList<Edge> labelLabelEdges;
		ArrayList<Edge> labelPredictionEdges;
		Edge edge;
		int vertex1, vertex2;
		double percentage;
		for (int i = 0; i < collections.size(); i++) {
			collection = collections.get(i);
			for (int j = 0; j < collection.examples.size(); j++) {
				example = collection.examples.get(j);
				for (int k = 0; k < example.denominatorExamples.size(); k++) {
					denominator = example.denominatorExamples.get(k);
					percentage = denominator.percentage;
					// wordLabelEdges;
					wordLabelEdges = denominator.wordLabelEdges;
					for (int w = 0; w < wordLabelEdges.size(); w++) {
						edge = wordLabelEdges.get(w);
						vertex1 = edge.vertex1;
						vertex2 = edge.vertex2;
						gradientWordIndexLabelIndexMatrix[vertex1][vertex2] -= percentage;
					}
					// labelPredictionEdges;
					labelPredictionEdges = denominator.labelPredictionEdges;
					for (int w = 0; w < labelPredictionEdges.size(); w++) {
						edge = labelPredictionEdges.get(w);
						vertex1 = edge.vertex1;
						vertex2 = edge.vertex2;
						gradientLabelIndexPredictedNodeMatrix[vertex1][vertex2] -= percentage;
					}
					// predictionObservationEdge;
					edge = denominator.predictionObservationEdge;
					vertex1 = edge.vertex1;
					vertex2 = edge.vertex2;
					gradientPredictedNodeObservedNodeMatrix[vertex1][vertex2] -= percentage;
					// labelLabelEdges;
					labelLabelEdges = denominator.labelLabelEdges;
					for (int w = 0; w < labelLabelEdges.size(); w++) {
						edge = labelLabelEdges.get(w);
						vertex1 = edge.vertex1;
						vertex2 = edge.vertex2;
						gradientLabelIndexLabelIndexMatrix[vertex1][vertex2] -= percentage;
						// label-label symmetry
						if (vertex1 != vertex2 && isLabelMatrixSymmetric)
							gradientLabelIndexLabelIndexMatrix[vertex2][vertex1] -= percentage;
					}

				}
			}
		}
	}

	private static void copyValuesFromAggregateMatricesToGradientMatrices() {
		for (int i = 0; i < numOfAllCUIs; i++) {
			for (int j = 0; j < numOfAbstractLabels; j++) {
				gradientWordIndexLabelIndexMatrix[i][j] = aggregateWordIndexLabelIndexMatrix[i][j];
			}
		}
		for (int i = 0; i < numOfAbstractLabels; i++) {
			for (int j = 0; j < numOfPredictedValues; j++) {
				gradientLabelIndexPredictedNodeMatrix[i][j] = aggregateLabelIndexPredictedNodeMatrix[i][j];
			}
		}
		// for (int i = 0; i < numOfAbstractLabels; i++) {
		// for (int j = 0; j < numOfAbstractLabels; j++) {
		// gradientLabelIndexLabelIndexMatrix[i][j] =
		// aggregateLabelIndexLabelIndexMatrix[i][j];
		// }
		// }
		if (isThereEdgeBetweenLabels) {
			for (int i = 0; i < numOfAbstractLabels; i++) {
				for (int j = 0; j < numOfAbstractLabels; j++) {
					gradientLabelIndexLabelIndexMatrix[i][j] = aggregateLabelIndexLabelIndexMatrix[i][j];
				}
			}
		}

		if (!isPredictionObservationWeightFixed) {
			for (int i = 0; i < numOfPredictedValues; i++) {
				for (int j = 0; j < numOfObservedValues; j++) {
					gradientPredictedNodeObservedNodeMatrix[i][j] = aggregatePredictedNodeObservedNodeMatrix[i][j];
				}
			}
		}

	}

	private static double getChangePercentage(double prev, double cur) {
		if (cur >= prev)
			return -Math.abs(cur - prev) / prev * 100;
		else
			return Math.abs(cur - prev) / prev * 100;
	}

	private static void setPercentageForDenominatorExample(
			ArrayList<TrainingExampleCollection> collections) {
		TrainingExampleCollection collection;
		ArrayList<DenominatorExample> denominators;
		DenominatorExample denominator;
		TrainingExample example;
		double[] exponents;
		double denomEquivalentExponent;
		for (int i = 0; i < collections.size(); i++) {
			collection = collections.get(i);
			for (int j = 0; j < collection.examples.size(); j++) {
				example = collection.examples.get(j);
				denominators = example.denominatorExamples;
				exponents = new double[denominators.size()];
				// modifies this part
				// + log{(Chain Length choose l)*(L-1)^l}
				for (int k = 0; k < denominators.size(); k++) {
					exponents[k] = denominators.get(k).totalWeights
							+ Math.log(CombinatoricsUtils.binomialCoefficient(
									example.chainLength, k)
									* Math.pow(numOfAbstractLabels - 1, k));
				}

				denomEquivalentExponent = computeEquivalentExponent(exponents);
				for (int k = 0; k < denominators.size(); k++) {
					denominator = denominators.get(k);
					denominator.percentage = Math.exp(exponents[k]
							- denomEquivalentExponent);
				}
			}
		}

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

	private static double computeEquivalentExponent(double[] exponents) {
		double equivalent;
		double max = maximum(exponents);
		if (max == Double.NEGATIVE_INFINITY)
			return Double.NEGATIVE_INFINITY;
		double sum = 0;
		for (int i = 0; i < exponents.length; i++) {
			if (exponents[i] - max > -20)
				sum += Math.exp(exponents[i] - max);
		}
		equivalent = max + Math.log(sum);
		return equivalent;
	}

	/**
	 * Returns Q function values with samples approximating expectation
	 * 
	 * @param collections
	 * @return
	 */
	// discarded with computing denominators exactly
	private static double getLogLikelihood(
			ArrayList<TrainingExampleCollection> collections) {
		TrainingExampleCollection collection;
		TrainingExample example;
		DenominatorExample edges;
		ArrayList<DenominatorExample> denominators;
		DenominatorExample denominator;
		int chainLength;
		double equivalentExponent;
		double[] exponents;
		double totalWeights = 0;
		for (int i = 0; i < collections.size(); i++) {
			collection = collections.get(i);
			for (int j = 0; j < collection.examples.size(); j++) {
				example = collection.examples.get(j);
				chainLength = example.chainLength;
				edges = getEdgesFromNodes(example);
				setTotalWeightsForDenomEx(edges);
				totalWeights += edges.totalWeights;
				/**********************************/
				denominators = example.denominatorExamples;
				exponents = new double[denominators.size()];
				for (int k = 0; k < denominators.size(); k++) {
					denominator = denominators.get(k);
					// ???
					exponents[k] = Math.log(CombinatoricsUtils
							.binomialCoefficient(chainLength, k))
							+ k
							* Math.log(numOfAbstractLabels - 1)
							+ denominator.totalWeights;
				}
				equivalentExponent = computeEquivalentExponent(exponents);
				totalWeights -= equivalentExponent;
				/**********************************/

			}
		}
		return totalWeights;
	}

	private static TrainingExampleCollection getTrainingExampleCollection(
			TrainingExampleCollection exampleColletion, int steps) {
		// System.out.println("Getting a colletion. Starts.");
		burnIn(steps, exampleColletion.examples);
		// System.out.println("Getting a collection. Terminates.");
		return exampleColletion.getDeepCopy();
	}

	private static void getStratifiedExamplesForExamples(
			ArrayList<TrainingExample> examples) {
		for (int i = 0; i < examples.size(); i++) {
			getStratifiedExamplesForSingleExample(examples.get(i));
		}
	}

	private static void getStratifiedExamplesForSingleExample(
			TrainingExample trainingExample) {
		trainingExample.denominatorExamples.clear();
		int chainLength = trainingExample.chainLength;
		// first 'chainLength' indices represent labels, then predicted node,
		// finallhy observed node
		int latentVars = chainLength + 1;
		if (isObservedVariableLatent)
			latentVars++;
		ArrayList<Integer> nodes = new ArrayList<Integer>();
		for (int i = 0; i < latentVars; i++) {
			nodes.add(i);
		}

		DenominatorExample denomExample;
		ArrayList<Integer> subset = new ArrayList<Integer>();

		// for (int i = 0; i < numOfStratifiedPartitionSamples; i++) {
		for (int i = 0; i <= chainLength; i++) {
			Collections.shuffle(nodes);
			subset.clear();
			for (int j = 0; j < i; j++) {
				try {
					subset.add(nodes.get(j));
				} catch (IndexOutOfBoundsException e) {
					System.err.println(e.getMessage());
				}

			}
			denomExample = getDenomExample(trainingExample.getDeepCopy(),
					subset);
			trainingExample.denominatorExamples.add(denomExample);
		}

	}

	private static DenominatorExample getDenomExample(
			TrainingExample trainingExample, ArrayList<Integer> indices) {

		int[] singletons = new int[numOfAbstractLabels];
		double[] probabilities = new double[numOfAbstractLabels];
		for (int i = 0; i < numOfAbstractLabels; i++) {
			singletons[i] = i;
			probabilities[i] = 1.0 / numOfAbstractLabels;
		}
		labelDistribution = new EnumeratedIntegerDistribution(singletons,
				probabilities);

		DenominatorExample ret;
		int index;
		int chainLength = trainingExample.chainLength;
		int originalValue, newValue;
		for (int i = 0; i < indices.size(); i++) {
			index = indices.get(i);
			if (index < chainLength) {
				originalValue = trainingExample.labels.get(index);
				newValue = labelDistribution.sample();
				while (originalValue == newValue) {
					newValue = labelDistribution.sample();
				}
				trainingExample.labels.set(index, newValue);
			} else if (index == chainLength) {
				originalValue = trainingExample.predictedRelation;
				newValue = 1 - originalValue;
				trainingExample.predictedRelation = newValue;
			} else if (index == chainLength + 1) {
				originalValue = trainingExample.observedRelation;
				newValue = 1 - originalValue;
				trainingExample.observedRelation = newValue;
			} else {

			}
		}
		ret = getEdgesFromNodes(trainingExample);
		setTotalWeightsForDenomEx(ret);
		return ret;
	}

	private static void setTotalWeightsForDenomEx(DenominatorExample denomEx) {
		double totalWeights = 0;
		int prediction = denomEx.predictionObservationEdge.vertex1;
		int observation = denomEx.predictionObservationEdge.vertex2;
		totalWeights += predictedNodeObservedNodeMatrix[prediction][observation];
		Edge edge1, edge2;
		int vertex1, vertex2;
		for (int i = 0; i < denomEx.wordLabelEdges.size(); i++) {
			edge1 = denomEx.wordLabelEdges.get(i);
			vertex1 = edge1.vertex1;
			vertex2 = edge1.vertex2;
			totalWeights += wordIndexLabelIndexMatrix[vertex1][vertex2];
			edge2 = denomEx.labelPredictionEdges.get(i);
			vertex1 = edge2.vertex1;
			vertex2 = edge2.vertex2;
			totalWeights += labelIndexPredictedNodeMatrix[vertex1][vertex2];
		}
		for (int i = 0; i < denomEx.labelLabelEdges.size(); i++) {
			edge1 = denomEx.labelLabelEdges.get(i);
			vertex1 = edge1.vertex1;
			vertex2 = edge1.vertex2;
			totalWeights += labelIndexLabelIndexMatrix[vertex1][vertex2];
		}
		denomEx.totalWeights = totalWeights;
		// denomEx.totalWeightsExp = Math.exp(totalWeights);
	}

	private static DenominatorExample getEdgesFromNodes(
			TrainingExample trainingExample) {
		DenominatorExample ret = new DenominatorExample();
		int observation = trainingExample.observedRelation;
		int prediction = trainingExample.predictedRelation;
		int label, wordIndex, prevLabel = -1;
		ret.predictionObservationEdge = new Edge(prediction, observation);
		for (int i = 0; i < trainingExample.chainLength; i++) {
			label = trainingExample.labels.get(i);
			wordIndex = trainingExample.wordIndices.get(i);
			ret.wordLabelEdges.add(new Edge(wordIndex, label));
			ret.labelPredictionEdges.add(new Edge(label, prediction));
			if (i != 0)
				ret.labelLabelEdges.add(new Edge(prevLabel, label));
			prevLabel = label;
		}
		return ret;
	}

	private static void assignAggregateMatrices(
			ArrayList<TrainingExample> examples) {
		TrainingExample example;
		ArrayList<Integer> wordIndices;
		ArrayList<Integer> labels;
		int observedRelation;
		int predictedRelation;
		int wordIndex;
		int label;
		int prevLabel = -1;

		for (int i = 0; i < examples.size(); i++) {
			example = examples.get(i);
			wordIndices = example.wordIndices;
			labels = example.labels;
			observedRelation = example.observedRelation;
			predictedRelation = example.predictedRelation;
			aggregatePredictedNodeObservedNodeMatrix[predictedRelation][observedRelation] += 1;
			for (int j = 0; j < wordIndices.size(); j++) {
				wordIndex = wordIndices.get(j);
				label = labels.get(j);
				aggregateWordIndexLabelIndexMatrix[wordIndex][label] += 1;
				aggregateLabelIndexPredictedNodeMatrix[label][predictedRelation] += 1;
				if (j != 0) {
					aggregateLabelIndexLabelIndexMatrix[prevLabel][label] += 1;
					// ??? label label symmetry
					if (prevLabel != label && isLabelMatrixSymmetric)
						aggregateLabelIndexLabelIndexMatrix[label][prevLabel] += 1;
				}
				prevLabel = label;
			}
		}

	}

	// aggregateWordIndexLabelIndexMatrix;
	// aggregateLabelIndexLabelIndexMatrix;
	// aggregateLabelIndexPredictedNodeMatrix;
	// aggregatePredictedNodeObservedNodeMatrix;

	private static void emptyAllAggregateMatrices() {
		// TODO Auto-generated method stub

		for (int i = 0; i < numOfAllCUIs; i++) {
			for (int j = 0; j < numOfAbstractLabels; j++) {
				aggregateWordIndexLabelIndexMatrix[i][j] = 0;
			}
		}

		for (int i = 0; i < numOfAbstractLabels; i++) {
			for (int j = 0; j < numOfAbstractLabels; j++) {
				aggregateLabelIndexLabelIndexMatrix[i][j] = 0;
			}
		}

		for (int i = 0; i < numOfAbstractLabels; i++) {
			for (int j = 0; j < numOfPredictedValues; j++) {
				aggregateLabelIndexPredictedNodeMatrix[i][j] = 0;
			}
		}

		for (int i = 0; i < numOfPredictedValues; i++) {
			for (int j = 0; j < numOfObservedValues; j++) {
				aggregatePredictedNodeObservedNodeMatrix[i][j] = 0;
			}
		}
	}

	private static void burnIn(int numOfSteps,
			ArrayList<TrainingExample> examples) {
		// TODO Auto-generated method stub
		for (int i = 0; i < numOfSteps; i++) {
			// System.out.println("burn-in step#" + i + " time(s): "
			// + System.currentTimeMillis() / 1000);
			updateExamplesUsingGibbsSampling(examples);
		}
	}

	/**
	 * Samples for labels first, the prediction node, then observation node.
	 * 
	 * @param examples
	 */
	private static void updateExamplesUsingGibbsSampling(
			ArrayList<TrainingExample> examples) {
		// TODO Auto-generated method stub
		TrainingExample example;
		int chainLength;
		EnumeratedIntegerDistribution dist;
		for (int i = 0; i < examples.size(); i++) {
			example = examples.get(i);
			chainLength = example.chainLength;
			// sampling for chain
			for (int j = 0; j < chainLength; j++) {
				if (j == 0)
					dist = getEnumeratedIntegerDistributionForChain("l",
							example.wordIndices.get(j), null,
							example.labels.get(j + 1),
							example.predictedRelation);
				else if (j == chainLength - 1)
					dist = getEnumeratedIntegerDistributionForChain("r",
							example.wordIndices.get(j),
							example.labels.get(j - 1), null,
							example.predictedRelation);
				else
					dist = getEnumeratedIntegerDistributionForChain("m",
							example.wordIndices.get(j),
							example.labels.get(j - 1),
							example.labels.get(j + 1),
							example.predictedRelation);
				example.labels.set(j, dist.sample());
			}
			// sampling for latent predicted node
			dist = getEnumeratedIntegerDistributionForPredictionNode(example);
			example.predictedRelation = dist.sample();
			if (isObservedVariableLatent) {
				dist = getEnumeratedIntegerDistributionForObservationNode(example);
				example.observedRelation = dist.sample();
			}
		}
	}

	private static EnumeratedIntegerDistribution getEnumeratedIntegerDistributionForObservationNode(
			TrainingExample example) {
		int[] singletons = new int[numOfObservedValues];
		double[] probabilities = new double[numOfObservedValues];
		double exponent;
		for (int i = 0; i < numOfObservedValues; i++) {
			singletons[i] = i;
			exponent = predictedNodeObservedNodeMatrix[example.predictedRelation][i];
			probabilities[i] = exponent;
		}
		double equivalentExponent = computeEquivalentExponent(probabilities);
		for (int i = 0; i < probabilities.length; i++) {
			probabilities[i] = Math.exp(probabilities[i] - equivalentExponent);
		}
		EnumeratedIntegerDistribution dist = new EnumeratedIntegerDistribution(
				singletons, probabilities);
		return dist;
	}

	private static EnumeratedIntegerDistribution getEnumeratedIntegerDistributionForPredictionNode(
			TrainingExample example) {
		// TODO Auto-generated method stub
		int[] singletons = new int[numOfPredictedValues];
		double[] probabilities = new double[numOfPredictedValues];
		double exponent;
		int label;
		for (int i = 0; i < numOfPredictedValues; i++) {
			exponent = 0;
			singletons[i] = i;
			for (int j = 0; j < example.labels.size(); j++) {
				label = example.labels.get(j);
				exponent += labelIndexPredictedNodeMatrix[label][i];
			}
			exponent += predictedNodeObservedNodeMatrix[i][example.observedRelation];
			probabilities[i] = exponent;
		}
		double equivalentExponent = computeEquivalentExponent(probabilities);
		for (int i = 0; i < probabilities.length; i++) {
			probabilities[i] = Math.exp(probabilities[i] - equivalentExponent);
		}
		EnumeratedIntegerDistribution dist = new EnumeratedIntegerDistribution(
				singletons, probabilities);
		return dist;
	}

	// EnumeratedIntegerDistribution(int[] singletons, double[] probabilities)
	private static EnumeratedIntegerDistribution getEnumeratedIntegerDistributionForChain(
			String position, Integer wordIndex, Integer leftLabelIndex,
			Integer rightLabelIndex, int predictedRelation) {
		// TODO Auto-generated method stub
		int[] singletons = new int[numOfAbstractLabels];
		double[] probabilities = new double[numOfAbstractLabels];

		for (int labelIndex = 0; labelIndex < numOfAbstractLabels; labelIndex++) {
			singletons[labelIndex] = labelIndex;
			if (position.equals("l")) {
				probabilities[labelIndex] = wordIndexLabelIndexMatrix[wordIndex][labelIndex]
						+ labelIndexLabelIndexMatrix[labelIndex][rightLabelIndex]
						+ labelIndexPredictedNodeMatrix[labelIndex][predictedRelation];
			} else if (position.equals("r")) {
				probabilities[labelIndex] = wordIndexLabelIndexMatrix[wordIndex][labelIndex]
						+ labelIndexLabelIndexMatrix[leftLabelIndex][labelIndex]
						+ labelIndexPredictedNodeMatrix[labelIndex][predictedRelation];
			} else {
				probabilities[labelIndex] = wordIndexLabelIndexMatrix[wordIndex][labelIndex]
						+ labelIndexLabelIndexMatrix[labelIndex][rightLabelIndex]
						+ labelIndexLabelIndexMatrix[leftLabelIndex][labelIndex]
						+ labelIndexPredictedNodeMatrix[labelIndex][predictedRelation];
			}
		}

		double equivalentExponent = computeEquivalentExponent(probabilities);
		for (int i = 0; i < probabilities.length; i++) {
			probabilities[i] = Math.exp(probabilities[i] - equivalentExponent);
		}

		EnumeratedIntegerDistribution dist = new EnumeratedIntegerDistribution(
				singletons, probabilities);
		return dist;
	}

	// fix this to make label by label matrix symmetric
	/**
	 * If isSymmetric is true, 2d array is symmetric.
	 * 
	 * @param isSymmetric
	 * @param d1
	 * @param d2
	 * @return
	 */
	private static double[][] makeMatrixWithInitialValues(boolean isSymmetric,
			int d1, int d2, boolean isEquallyWeighted) {
		// TODO Auto-generated method stub
		double[][] matrix = new double[d1][d2];
		GammaDistribution dist = new GammaDistribution(1, 1);
		if (!isSymmetric) {
			for (int i = 0; i < d1; i++) {
				for (int j = 0; j < d2; j++) {
					if (isEquallyWeighted)
						matrix[i][j] = 1;
					else
						matrix[i][j] = dist.sample();
				}
			}
		} else {
			for (int i = 0; i < d1; i++) {
				for (int j = 0; j <= i; j++) {
					// label label symmetry
					if (isEquallyWeighted)
						matrix[i][j] = 1;
					else
						matrix[i][j] = dist.sample();
					matrix[j][i] = matrix[i][j];
				}
			}
		}

		return matrix;
	}

	private static void assignInitialValuesForLatentVariables(
			ArrayList<TrainingExample> examples, int numberOfAbstractLabels) {
		// TODO Auto-generated method stub
		UniformIntegerDistribution dist = new UniformIntegerDistribution(0,
				numberOfAbstractLabels - 1);
		UniformIntegerDistribution obDist = new UniformIntegerDistribution(0,
				numOfObservedValues - 1);
		UniformIntegerDistribution predDist = new UniformIntegerDistribution(0,
				numOfPredictedValues - 1);
		int num;
		TrainingExample example;
		for (int i = 0; i < examples.size(); i++) {
			example = examples.get(i);
			num = predDist.sample();
			if (initialPredictionObservationEquality)
				example.predictedRelation = example.observedRelation;
			else
				example.predictedRelation = predDist.sample();
			if (isObservedVariableLatent) {
				num = obDist.sample();
				example.observedRelation = num;
			}
			for (int j = 0; j < example.chainLength; j++) {
				if (labelWordsInBetweenZero && example.observedRelation == 1) {
					if (j > example.relationWordIndex1
							&& j < example.relationWordIndex2)
						example.labels.add(0);
					else
						example.labels.add(dist.sample());
				} else
					example.labels.add(dist.sample());
			}

		}
	}

	private static void labelObservedRelationNode(
			ArrayList<TrainingExample> examples,
			Map<String, ArrayList<String>> relationMap) {
		// TODO Auto-generated method stub
		TrainingExample example;
		String relationWord1, word;
		int chainLength;
		for (int i = 0; i < examples.size(); i++) {
			example = examples.get(i);
			chainLength = example.wordIndices.size();
			example.chainLength = chainLength;
			relationWord1 = example.words.get(example.relationWordIndex1);
			for (int j = 0; j < chainLength; j++) {
				word = example.words.get(j);
				if (relationMap.get(relationWord1) != null
						&& relationMap.get(relationWord1).contains(word)) {
					example.observedRelation = 1;
					treatRelation++;
					break;
				}
			}
			example.words.clear();

		}

	}

	private static Map<String, ArrayList<String>> getRelationMatchings()
			throws Exception {
		Map<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();
		BufferedReader br = new BufferedReader(new FileReader(
				"formatted_relation_file"));
		String line;
		String[] CUIs;
		while ((line = br.readLine()) != null) {
			CUIs = line.split("\\s");
			if (map.get(CUIs[0]) == null) {
				ArrayList<String> list = new ArrayList<String>();
				list.add(CUIs[1]);
				map.put(CUIs[0], list);
			} else {
				map.get(CUIs[0]).add(CUIs[1]);
			}
		}
		return map;
	}

	/**
	 * puts 20 words before disease if not limited, puts 9 words after disease
	 * if not limited
	 * 
	 * @param CUIs
	 * @param CUIIndices
	 * @param numIndices
	 * @return
	 */
	private static ArrayList<TrainingExample> makeTrainingExamples(
			String[] CUIs, int[] CUIIndices, int[] numIndices) {
		if (numIndices == null)
			return new ArrayList<TrainingExample>();
		else {
			ArrayList<TrainingExample> ret = new ArrayList<TrainingExample>();
			int index;

			for (int i = 0; i < numIndices.length; i++) {
				TrainingExample example = new TrainingExample();
				index = numIndices[i];
				for (int j = (index >= 20) ? index - 20 : 0; j < (index + 10 <= CUIIndices.length ? index + 10
						: CUIIndices.length); j++) {
					example.wordIndices.add(CUIIndices[j]);
					example.words.add(CUIs[j]);
					if (j == index)
						example.relationWordIndex1 = example.wordIndices.size() - 1;
				}
				example.chainLength = example.wordIndices.size();
				ret.add(example);
			}
			return ret;
		}
	}

	private static void pause() throws InterruptedException {
		// Scanner input = new Scanner(System.in);
		// System.out.print("Press enter to exit....");
		// input.hasNextLine();
		// Thread.sleep(4000);
	}

}

// Termination condition 1
// if (Math.abs((curMaximizationLikelihood -
// prevMaximizationLikelihood)
// / prevMaximizationLikelihood) <
// maximizationTerminalPercentage) {
// System.out
// .println("Percentage change is too small. Break.");
// break;
// }

// Termination condition 2
// if (MaxIter % maximizationPrintoutInterval == 0) {
// breakFlag = setMaxPhase(prevMaximizationLikelihood,
// curMaximizationLikelihood);
// System.out.println("current phase is " + phases[maxPhase]);
// // if (percentage < 1.0 / 100)
// // breakFlag = true;
// // if (change < 0.01)
// // breakFlag = true;
// if (breakFlag) {
// System.out.println("finished three-step phase");
// break;
// }
//
// }

// Termination condition 3
// if (curMaximizationLikelihood < prevMaximizationLikelihood) {
// System.out.println("Likelihood is decreasing. Break.");
// break;
// }

// Customizes prediction-observation matrix
// predictedNodeObservedNodeMatrix[0][0] = 100;
// predictedNodeObservedNodeMatrix[0][1] = -100;
// predictedNodeObservedNodeMatrix[1][0] = -100;
// predictedNodeObservedNodeMatrix[1][1] = 100;

// wordIndexLabelIndexMatrix[0][0] = 1;
// wordIndexLabelIndexMatrix[0][1] = 0;
// wordIndexLabelIndexMatrix[1][0] = 0;
// wordIndexLabelIndexMatrix[1][1] = 1;
//
// labelIndexPredictedNodeMatrix[0][0] = 0.3;
// labelIndexPredictedNodeMatrix[0][1] = 1;
// labelIndexPredictedNodeMatrix[1][0] = 1;
// labelIndexPredictedNodeMatrix[1][1] = 0.3;

// String[] phases = { "initial", "first increasing phase",
// "first decreasing phase", "second increasing phase",
// "closing phase" };

// Prepares distributions
// int[] singletons = new int[numOfAbstractLabels];
// double[] probabilities = new double[numOfAbstractLabels];
// for (int i = 0; i < numOfAbstractLabels; i++) {
// singletons[i] = i;
// probabilities[i] = 1.0 / numOfAbstractLabels;
// }
// labelDistribution = new EnumeratedIntegerDistribution(singletons,
// probabilities);
//
// singletons = new int[numOfPredictedValues];
// probabilities = new double[numOfPredictedValues];
// for (int i = 0; i < numOfPredictedValues; i++) {
// singletons[i] = i;
// probabilities[i] = 1.0 / numOfPredictedValues;
// }
// predictionDistribution = new
// EnumeratedIntegerDistribution(singletons,
// probabilities);
//
// singletons = new int[numOfObservedValues];
// probabilities = new double[numOfObservedValues];
// for (int i = 0; i < numOfObservedValues; i++) {
// singletons[i] = i;
// probabilities[i] = 1.0 / numOfObservedValues;
// }
// observationDistribution = new
// EnumeratedIntegerDistribution(singletons,
// probabilities);
