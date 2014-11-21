package part1;

public class MRRELRecord {
	public int index;
	public String[] items;

	public String getCUI1() {
		return items[0];
	}

	public String getAUI1() {
		return items[1];
	}

	public String getCUI2() {
		return items[4];
	}

	public String getAUI2() {
		return items[5];
	}

	public String getREL() {
		return items[3];
	}

	public String getRELA() {
		return items[7];
	}
}
