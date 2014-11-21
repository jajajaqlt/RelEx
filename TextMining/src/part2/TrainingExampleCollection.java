package part2;

import java.util.ArrayList;

public class TrainingExampleCollection {
	public ArrayList<TrainingExample> examples;

	public TrainingExampleCollection() {
		examples = new ArrayList<TrainingExample>();
	}

	public TrainingExampleCollection getDeepCopy() {
		TrainingExampleCollection ret = new TrainingExampleCollection();
		for (int i = 0; i < examples.size(); i++) {
			ret.examples.add(this.examples.get(i).getDeepCopy());
		}
		return ret;
	}
}
