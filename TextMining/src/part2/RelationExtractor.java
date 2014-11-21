package part2;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;
import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.distribution.UniformIntegerDistribution;

public class RelationExtractor {

	public static double[][] wordIndexLabelIndexMatrix;
	public static double[][] labelIndexLabelIndexMatrix;
	public static double[][] labelIndexPredictedNodeMatrix;
	public static double[][] predictedNodeObservedNodeMatrix;

	public static double[][] aggregateWordIndexLabelIndexMatrix;
	public static double[][] aggregateLabelIndexLabelIndexMatrix;
	public static double[][] aggregateLabelIndexPredictedNodeMatrix;
	public static double[][] aggregatePredictedNodeObservedNodeMatrix;

	public static double[][] gradientWordIndexLabelIndexMatrix;
	public static double[][] gradientLabelIndexLabelIndexMatrix;
	public static double[][] gradientLabelIndexPredictedNodeMatrix;
	public static double[][] gradientPredictedNodeObservedNodeMatrix;

	public static int numOfAllCUIs;
	public static int numOfAbstractLabels;
	public static int numOfPredictedValues;
	public static int numOfObservedValues;

	public static int numOfExpectationSamples;
	public static int numOfStratifiedPartitionSamples;

	public static int latentObservedNode = 0;

	public static EnumeratedIntegerDistribution labelDistribution;
	public static EnumeratedIntegerDistribution predictionDistribution;

	public static double stepSize = 0.1;

	public static void main(String[] args) throws Exception {
		// break point to test assignment of instance variables
		BufferedReader br = new BufferedReader(new FileReader(
				"formatted_output_test"));
		String line, firstLine;
		boolean isCUILine = true;
		String[] CUIs = null, strIndices, allCUIs;
		int[] numIndices, CUIIndices = null;
		Map<String, Integer> CUIIndexMap = new HashMap<String, Integer>();
		Map<Integer, String> indexCUIMap = new HashMap<Integer, String>();
		ArrayList<TrainingExample> examples = new ArrayList<TrainingExample>();
		// numOfAbstractLabels = 10;
		numOfAbstractLabels = 2;
		numOfPredictedValues = 2;
		numOfObservedValues = 2;
		// numOfExpectationSamples = 50;
		numOfExpectationSamples = 2;
		// numOfStratifiedPartitionSamples = 10;
		numOfStratifiedPartitionSamples = 2;

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
		Map<String, ArrayList<String>> relationMap = getRelationMatchings();
		labelObservedRelationNode(examples, relationMap);
		assignInitialValuesForLatentVariables(examples, numOfAbstractLabels);
		// break point to test accuracy of examples
		// matrices
		wordIndexLabelIndexMatrix = makeMatrixWithInitialValues(false,
				numOfAllCUIs, numOfAbstractLabels);
		labelIndexLabelIndexMatrix = makeMatrixWithInitialValues(true,
				numOfAbstractLabels, numOfAbstractLabels);
		labelIndexPredictedNodeMatrix = makeMatrixWithInitialValues(false,
				numOfAbstractLabels, numOfPredictedValues);
		predictedNodeObservedNodeMatrix = makeMatrixWithInitialValues(false,
				numOfPredictedValues, numOfObservedValues);

		aggregateWordIndexLabelIndexMatrix = new double[numOfAllCUIs][numOfAbstractLabels];
		aggregateLabelIndexLabelIndexMatrix = new double[numOfAbstractLabels][numOfAbstractLabels];
		aggregateLabelIndexPredictedNodeMatrix = new double[numOfAbstractLabels][numOfPredictedValues];
		aggregatePredictedNodeObservedNodeMatrix = new double[numOfPredictedValues][numOfObservedValues];

		gradientWordIndexLabelIndexMatrix = new double[numOfAllCUIs][numOfAbstractLabels];
		gradientLabelIndexLabelIndexMatrix = new double[numOfAbstractLabels][numOfAbstractLabels];
		gradientLabelIndexPredictedNodeMatrix = new double[numOfAbstractLabels][numOfPredictedValues];
		gradientPredictedNodeObservedNodeMatrix = new double[numOfPredictedValues][numOfObservedValues];

		// wordIndexLabelIndexMatrix,labelIndexLabelIndexMatrix,labelIndexPredictedNodeMatrix,predictedNodeObservedNodeMatrix
		// break point to test burn-in process
		burnIn(1000, examples);
		TrainingExampleCollection exampleColletion = new TrainingExampleCollection();
		exampleColletion.examples = examples;
		ArrayList<TrainingExampleCollection> collections = new ArrayList<TrainingExampleCollection>();
		TrainingExampleCollection temp;

		int[] singletons = new int[numOfAbstractLabels];
		double[] probabilities = new double[numOfAbstractLabels];
		for (int i = 0; i < numOfAbstractLabels; i++) {
			singletons[i] = i;
			probabilities[i] = 1.0 / numOfAbstractLabels;
		}
		labelDistribution = new EnumeratedIntegerDistribution(singletons,
				probabilities);

		singletons = new int[numOfPredictedValues];
		probabilities = new double[numOfPredictedValues];
		for (int i = 0; i < numOfPredictedValues; i++) {
			singletons[i] = i;
			probabilities[i] = 1.0 / numOfPredictedValues;
		}
		predictionDistribution = new EnumeratedIntegerDistribution(singletons,
				probabilities);

		double prevExpectationLikelihood = 0, curExpectationLikelihood = 2, prevMaximizationLikelihood, curMaximizationLikelihood = 0;
		// EM algorithm starts here
		while (getChangePercentage(prevExpectationLikelihood,
				curExpectationLikelihood) > 1) {
			// Expectation
			// get 'numOfExpectationSamples' samples
			// burn in for another 500 times
			collections.clear();
			emptyAllAggregateMatrices();
			burnIn(500, exampleColletion.examples);
			prevExpectationLikelihood = curExpectationLikelihood;
			for (int i = 0; i < numOfExpectationSamples; i++) {
				temp = getTrainingExampleCollection(exampleColletion, 100);
				collections.add(temp);
				assignAggregateMatrices(temp.examples);
				getStratifiedExamplesForExamples(temp.examples);
			}
			setPercentageForDenominatorExample(collections);
			// !!! setPercentageForDenominatorExample(collections);
			// curExpectationLikelihood =
			// getTotalWeightsFromCollections(collections);
			// Maximization
			// do gradient matrices
			// update weights and total weights iteratively until no likelihood
			// increase

			// break point to test stratified examples (including percentage)
			// and aggregate matrices
			curMaximizationLikelihood = getTotalWeightsFromCollections(collections);
			prevMaximizationLikelihood = 0;
			while (getChangePercentage(prevMaximizationLikelihood,
					curMaximizationLikelihood) > 1) {
				prevMaximizationLikelihood = curMaximizationLikelihood;
				// 1. calculate gradients
				// 2. update weight matrices
				// 3. recalculate weights for each denominator example
				calculateGradients(collections);
				updateWeightMatricesFromGradientMatrices();
				curMaximizationLikelihood = getTotalWeightsFromCollections(collections);
				setPercentageForDenominatorExample(collections);
				// break point to test gradient matrices and get-total-weights
				// function
			}

			boolean flag = true;
			flag = false;

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
						if (vertex1 != vertex2)
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
		for (int i = 0; i < numOfAbstractLabels; i++) {
			for (int j = 0; j < numOfAbstractLabels; j++) {
				gradientLabelIndexLabelIndexMatrix[i][j] = aggregateLabelIndexLabelIndexMatrix[i][j];
			}
		}
		for (int i = 0; i < numOfPredictedValues; i++) {
			for (int j = 0; j < numOfObservedValues; j++) {
				gradientPredictedNodeObservedNodeMatrix[i][j] = aggregatePredictedNodeObservedNodeMatrix[i][j];
			}
		}

	}

	private static double getChangePercentage(double prev, double cur) {
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
				for (int k = 0; k < denominators.size(); k++) {
					exponents[k] = denominators.get(k).totalWeights;
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
			sum += Math.exp(exponents[i] - max);
		}
		equivalent = max + Math.log(sum);
		return equivalent;
	}

	private static double getTotalWeightsFromCollections(
			ArrayList<TrainingExampleCollection> collections) {
		TrainingExampleCollection collection;
		TrainingExample example;
		DenominatorExample edges;
		double totalWeights = 0;
		for (int i = 0; i < collections.size(); i++) {
			collection = collections.get(i);
			for (int j = 0; j < collection.examples.size(); j++) {
				example = collection.examples.get(j);
				edges = getEdgesFromNodes(example);
				setTotalWeightsForDenomEx(edges);
				totalWeights += edges.totalWeights;
			}
		}
		return totalWeights;
	}

	private static TrainingExampleCollection getTrainingExampleCollection(
			TrainingExampleCollection exampleColletion, int steps) {
		burnIn(steps, exampleColletion.examples);
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
		int latentVars = chainLength + 1 + latentObservedNode;
		ArrayList<Integer> nodes = new ArrayList<Integer>();
		for (int i = 0; i < latentVars; i++) {
			nodes.add(i);
		}

		DenominatorExample denomExample;
		ArrayList<Integer> subset = new ArrayList<Integer>();

		for (int i = 0; i < numOfStratifiedPartitionSamples; i++) {
			Collections.shuffle(nodes);
			subset.clear();
			for (int j = 0; j < i; j++) {
				subset.add(nodes.get(j));
			}
			denomExample = getDenomExample(trainingExample.getDeepCopy(),
					subset);
			trainingExample.denominatorExamples.add(denomExample);
		}

	}

	private static DenominatorExample getDenomExample(
			TrainingExample trainingExample, ArrayList<Integer> indices) {
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
					if (prevLabel != label)
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
			updateExamplesUsingGibbsSampling(examples);
		}
	}

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
		}
	}

	private static EnumeratedIntegerDistribution getEnumeratedIntegerDistributionForPredictionNode(
			TrainingExample example) {
		// TODO Auto-generated method stub
		int[] singletons = new int[numOfPredictedValues];
		double[] probabilities = new double[numOfPredictedValues];
		double exponent = 0;
		int label;
		for (int i = 0; i < numOfPredictedValues; i++) {
			singletons[i] = i;
			for (int j = 0; j < example.labels.size(); j++) {
				label = example.labels.get(j);
				exponent += labelIndexPredictedNodeMatrix[label][i];
			}
			exponent += predictedNodeObservedNodeMatrix[i][example.observedRelation];
			probabilities[i] = Math.exp(exponent);
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
				probabilities[labelIndex] = Math
						.exp(wordIndexLabelIndexMatrix[wordIndex][labelIndex]
								+ labelIndexLabelIndexMatrix[labelIndex][rightLabelIndex]
								+ labelIndexPredictedNodeMatrix[labelIndex][predictedRelation]);
			} else if (position.equals("r")) {
				probabilities[labelIndex] = Math
						.exp(wordIndexLabelIndexMatrix[wordIndex][labelIndex]
								+ labelIndexLabelIndexMatrix[leftLabelIndex][labelIndex]
								+ labelIndexPredictedNodeMatrix[labelIndex][predictedRelation]);
			} else {
				probabilities[labelIndex] = Math
						.exp(wordIndexLabelIndexMatrix[wordIndex][labelIndex]
								+ labelIndexLabelIndexMatrix[labelIndex][rightLabelIndex]
								+ labelIndexLabelIndexMatrix[leftLabelIndex][labelIndex]
								+ labelIndexPredictedNodeMatrix[labelIndex][predictedRelation]);
			}
		}

		EnumeratedIntegerDistribution dist = new EnumeratedIntegerDistribution(
				singletons, probabilities);
		return dist;
	}

	// fix this to make label by label matrix symmetric
	private static double[][] makeMatrixWithInitialValues(boolean isSymmetric,
			int d1, int d2) {
		// TODO Auto-generated method stub
		double[][] matrix = new double[d1][d2];
		GammaDistribution dist = new GammaDistribution(1, 1);
		if (!isSymmetric) {
			for (int i = 0; i < d1; i++) {
				for (int j = 0; j < d2; j++) {
					matrix[i][j] = dist.sample();
				}
			}
		} else {
			for (int i = 0; i < d1; i++) {
				for (int j = 0; j <= i; j++) {
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
		UniformIntegerDistribution binDist = new UniformIntegerDistribution(0,
				1);
		int num;
		TrainingExample example;
		for (int i = 0; i < examples.size(); i++) {
			example = examples.get(i);
			num = binDist.sample();
			example.predictedRelation = num;
			for (int j = 0; j < example.chainLength; j++) {
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
				ret.add(example);
			}
			return ret;
		}
	}

}
