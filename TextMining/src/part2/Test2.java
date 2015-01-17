package part2;

import java.util.ArrayList;

public class Test2 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// double[] arr = {3,3.1};
		// System.out.println(computeEquivalentExponent(arr));
		// System.out.println(Math.exp(Double.NEGATIVE_INFINITY));
		TrainingExample ex = new TrainingExample();
		ex.chainLength = 10;
		
		ex.wordIndices.add(0);
		ex.wordIndices.add(1);
		ex.wordIndices.add(0);
		ex.wordIndices.add(1);
		ex.wordIndices.add(0);
		ex.wordIndices.add(1);
		ex.wordIndices.add(0);
		ex.wordIndices.add(1);
		ex.wordIndices.add(0);
		ex.wordIndices.add(1);
		
		ArrayList<TrainingExample> arr = new ArrayList<TrainingExample>();
		arr.add(ex);
		assignDistinctWordIndicesField(arr);
		System.out.println("Hello World");
		System.out.println("Hello SJZ");
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
}
