package part2;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.math3.distribution.EnumeratedIntegerDistribution;

/**
 * 3 words before, word1, words between, word2, 3 words after
 * relation_indicator, word1_index, word2_index
 * 
 * @author letaoqi
 *
 */
public class GeneratingSyntheticData {

	// public int dictSize = 1000;
	// public int docSize = 150000;
	// public int sampleInterval = 50;
	public static void main(String[] args) throws Exception {
		/**
		 * 
		 * @param fName
		 *            : synthetic
		 * @param dictSize
		 *            : 1000
		 * @param docSize
		 *            : 150000
		 * @param sampleInterval
		 *            : 50
		 * @param leftWordsSize
		 *            : 3
		 * @param rightWordsSize
		 *            : 3
		 * @param relationProb
		 *            : 0.1
		 * @param wordsChangeProb
		 *            : 0.5
		 * @param keyWordsSize
		 *            : 10
		 */
		/**
		 * 
		 * @param fName: synthetic#
		 * @param dictSize: 20
		 * @param docSize: 1000
		 * @param sampleInterval: 50
		 * @param leftWordsSize: # of words before left key word 3
		 * @param rightWordsSize: # of words after right key word 3
		 * @param relationProb: probability of on sample being positive 0.5
		 * @param wordsChangePercent: percentage of words between key words being changed to key words
		 * @param keyWordsSize: # of key words in dict
		 * @throws Exception
		 */
		new GeneratingSyntheticData("synthetic12", 20, 1000, 50, 3, 3, 0.5,
				1, 2);
		/**
		 * 
		 * @param testFName: synthetic#_test
		 * @param dictSize: 20
		 * @param leftWordsSize: # of words before left key word 3
		 * @param rightWordsSize: # of words after right key word 3
		 * @param keyWordsSize: # of key words in dict
		 * @param subChainLength: used, could be any number
		 * @param sizeForEachType: number of samples for each type
		 * @param sampleInterval: 50
		 * @throws Exception
		 */

//		GeneratingSyntheticData g = new GeneratingSyntheticData();
//		g.GeneratingSyntheticTestingExamples("synthetic11_test", 20, 3, 3, 2,
//				10, 2 * 8, 50);

	}

	public GeneratingSyntheticData() {

	}

	/**
	 * 
	 * @param fName
	 *            : synthetic
	 * @param dictSize
	 *            : 1000
	 * @param docSize
	 *            : 150000
	 * @param sampleInterval
	 *            : 50
	 * @param leftWordsSize
	 *            : 3
	 * @param rightWordsSize
	 *            : 3
	 * @param relationProb
	 *            : 0.1
	 * @param wordsChangeProb
	 *            : 0.5
	 * @param keyWordsSize
	 *            : 10
	 */
	public GeneratingSyntheticData(String fName, int dictSize, int docSize,
			int sampleInterval, int leftWordsSize, int rightWordsSize,
			double relationProb, double wordsChangePercent, int keyWordsSize)
			throws Exception {
		BufferedWriter bw = new BufferedWriter(new FileWriter(fName));
		int[] singletons = new int[dictSize];
		double[] probabilities = new double[dictSize];
		for (int i = 0; i < dictSize; i++) {
			singletons[i] = i;
			probabilities[i] = 1.0 / dictSize;
		}
		EnumeratedIntegerDistribution dictDist = new EnumeratedIntegerDistribution(
				singletons, probabilities);
		int examplesSize = docSize / sampleInterval;
		int[] exampleWords = new int[sampleInterval];
		singletons = new int[2];
		probabilities = new double[2];
		singletons[0] = 0;
		probabilities[0] = 1 - relationProb;
		singletons[1] = 1;
		probabilities[1] = relationProb;
		EnumeratedIntegerDistribution relationDist = new EnumeratedIntegerDistribution(
				singletons, probabilities);
		singletons = new int[sampleInterval];
		probabilities = new double[sampleInterval];
		for (int i = 0; i < sampleInterval; i++) {
			singletons[i] = i;
			probabilities[i] = 1.0 / sampleInterval;
		}
		EnumeratedIntegerDistribution wordIndexDist = new EnumeratedIntegerDistribution(
				singletons, probabilities);
		singletons = new int[keyWordsSize];
		probabilities = new double[keyWordsSize];
		for (int i = 0; i < keyWordsSize; i++) {
			singletons[i] = i;
			probabilities[i] = 1.0 / keyWordsSize;
		}
		EnumeratedIntegerDistribution keyWordsDist = new EnumeratedIntegerDistribution(
				singletons, probabilities);
		int relationIndicator, word1, word2, temp, startIndex, endIndex, wordIndex;
		double gap;
		ArrayList<Integer> innerWords = new ArrayList<Integer>();
		for (int i = 0; i < examplesSize; i++) {
			innerWords.clear();
			for (int j = 0; j < sampleInterval; j++) {
				exampleWords[j] = dictDist.sample();
			}
			word1 = wordIndexDist.sample();
			word2 = wordIndexDist.sample();
			while (Math.abs(word2 - word1) <= 1)
				word2 = wordIndexDist.sample();
			if (word1 > word2) {
				temp = word1;
				word1 = word2;
				word2 = temp;
			}
			startIndex = word1 - leftWordsSize;
			startIndex = startIndex < 0 ? 0 : startIndex;
			endIndex = word2 + rightWordsSize;
			endIndex = endIndex >= sampleInterval ? sampleInterval - 1
					: endIndex;
			relationIndicator = relationDist.sample();
			if (relationIndicator == 1) {
				gap = Math.ceil((word2 - word1 - 1) * wordsChangePercent);
				for (int j = word1 + 1; j < word2; j++) {
					innerWords.add(j);
				}
				Collections.shuffle(innerWords);
				for (int j = 0; j < gap; j++) {
					wordIndex = innerWords.get(j);
					exampleWords[wordIndex] = keyWordsDist.sample();
				}
			}
			for (int j = startIndex; j <= endIndex; j++) {
				bw.write("" + exampleWords[j]);
				bw.write(" ");
			}
			bw.newLine();
			bw.write("" + relationIndicator + " ");
			// bw.write("" + word1 + " ");
			// bw.write("" + word2 + " ");
			bw.write("" + leftWordsSize + " ");
			bw.write("" + (leftWordsSize + word2 - word1) + " ");
			bw.newLine();
		}
		bw.close();

	}

/**
 * 
 * @param testFName
 * @param dictSize
 * @param leftWordsSize
 * @param rightWordsSize
 * @param keyWordsSize
 * @param subChainLength
 * @param sizeForEachType
 * @param sampleInterval
 * @throws Exception
 */
	private void GeneratingSyntheticTestingExamples(String testFName,
			int dictSize, int leftWordsSize, int rightWordsSize,
			int keyWordsSize, int subChainLength, int sizeForEachType,
			int sampleInterval) throws Exception {
		BufferedWriter bw = new BufferedWriter(new FileWriter(testFName));
		int[] singletons = new int[dictSize];
		double[] probabilities = new double[dictSize];
		for (int i = 0; i < dictSize; i++) {
			singletons[i] = i;
			probabilities[i] = 1.0 / dictSize;
		}
		EnumeratedIntegerDistribution dictDist = new EnumeratedIntegerDistribution(
				singletons, probabilities);
		singletons = new int[2];
		probabilities = new double[2];
		singletons = new int[keyWordsSize];
		probabilities = new double[keyWordsSize];
		for (int i = 0; i < keyWordsSize; i++) {
			singletons[i] = i;
			probabilities[i] = 1.0 / keyWordsSize;
		}
		EnumeratedIntegerDistribution keyWordsDist = new EnumeratedIntegerDistribution(
				singletons, probabilities);

		// sizes of testing examples of different types
		int initialSize = sizeForEachType;

		int top0Bot0Uniform = initialSize, top0Bot0Random = initialSize;
		int top1Bot0Uniform = initialSize, top1Bot0Random = initialSize;
		int top0Bot1Uniform = initialSize, top0Bot1Random = initialSize;
		int top1Bot1Uniform = initialSize, top1Bot1Random = initialSize;

		// top 0 bottom 0 examples
		// addTestingSamples(top0Bot0Uniform, bw, dictDist, leftWordsSize,
		// rightWordsSize, keyWordsSize, subChainLength, true,
		// keyWordsDist, 0, true);

		addTestingSamples(top0Bot0Random, bw, dictDist, leftWordsSize,
				rightWordsSize, keyWordsSize, subChainLength, false,
				keyWordsDist, 0, true, sampleInterval);

		// top 1 bottom 0 examples
		// addTestingSamples(top1Bot0Uniform, bw, dictDist, leftWordsSize,
		// rightWordsSize, keyWordsSize, subChainLength, true,
		// keyWordsDist, 1, true);

		addTestingSamples(top1Bot0Random, bw, dictDist, leftWordsSize,
				rightWordsSize, keyWordsSize, subChainLength, false,
				keyWordsDist, 1, true, sampleInterval);

		// top 0 bottom 1 examples
		/**
		 * Generating testing examples of type 1. Type 1 is like this "165 391
		 * 926 1 1 1 1 1 1 1 1 1 1 310 28 577 0 10 19"
		 */
		// addTestingSamples(top0Bot1Uniform, bw, dictDist, leftWordsSize,
		// rightWordsSize, keyWordsSize, subChainLength, true,
		// keyWordsDist, 0, false);

		/**
		 * Generating testing examples of type 1. Type 2: words in between word1
		 * and word2 are sampled from key word distribution.
		 */
		addTestingSamples(top0Bot1Random, bw, dictDist, leftWordsSize,
				rightWordsSize, keyWordsSize, subChainLength, false,
				keyWordsDist, 0, false, sampleInterval);

		// top 1 bottom 1 examples
		// addTestingSamples(top1Bot1Uniform, bw, dictDist, leftWordsSize,
		// rightWordsSize, keyWordsSize, subChainLength, true,
		// keyWordsDist, 1, false);

		addTestingSamples(top1Bot1Random, bw, dictDist, leftWordsSize,
				rightWordsSize, keyWordsSize, subChainLength, false,
				keyWordsDist, 1, false, sampleInterval);

		bw.close();
	}

	private void addTestingSamples(int testSamplesSize, BufferedWriter bw,
			EnumeratedIntegerDistribution dictDist, int leftWordsSize,
			int rightWordsSize, int keyWordsSize, int keywordChainLength,
			boolean isUniform, EnumeratedIntegerDistribution keyWordsDist,
			int observationVal, boolean isRandomChain, int sampleInterval)
			throws Exception {

		EnumeratedIntegerDistribution intervalSizeDist;
		int trimmedIntervalSize = sampleInterval - leftWordsSize - rightWordsSize - 2;
		int[] singletons = new int[trimmedIntervalSize];
		double[] probabilities = new double[trimmedIntervalSize];
		int index = 0;
		for (int i = 1; i <= trimmedIntervalSize; i++) {
			singletons[i-1] = i;
			probabilities[i-1] = 1.0 / trimmedIntervalSize;
		}
		intervalSizeDist = new EnumeratedIntegerDistribution(singletons, probabilities);
		
		for (int i = 0; i < testSamplesSize; i++) {
			for (int j = 0; j < leftWordsSize + 1; j++) {
				bw.write("" + dictDist.sample() + " ");
			}
			
			keywordChainLength = intervalSizeDist.sample();
			for (int j = 0; j < keywordChainLength; j++) {
				if (!isRandomChain) {
					if (isUniform)
						bw.write("" + (i % keyWordsSize) + " ");
					else
						bw.write("" + keyWordsDist.sample() + " ");
				} else {
					bw.write("" + dictDist.sample() + " ");
				}

			}

			for (int j = 0; j < rightWordsSize + 1; j++) {
				bw.write("" + dictDist.sample() + " ");
			}

			bw.newLine();
			// relation indicator, i.e. observation corresponding prediction
			// should be declared 1
			// bw.write("" + 0 + " ");
			bw.write("" + observationVal + " ");
			bw.write("" + leftWordsSize + " ");
			bw.write("" + (leftWordsSize + keywordChainLength + 1) + " ");
			bw.newLine();
		}

	}

	public int countDictSize(String fName) throws Exception {
		int ret;
		BufferedReader br = new BufferedReader(new FileReader(fName));
		String line;
		String[] words;
		int word;
		Set<Integer> set = new HashSet<Integer>();
		while ((line = br.readLine()) != null) {
			words = line.split("\\s");
			for (int i = 0; i < words.length; i++) {
				word = Integer.parseInt(words[i]);
				set.add(word);
			}
			br.readLine();
		}
		br.close();
		ret = set.size();
		return ret;
	}

}
