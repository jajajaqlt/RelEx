package part2;

import java.io.BufferedReader; 
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class FilterOutTuplesWithTreatRelations {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		BufferedReader br = new BufferedReader(new FileReader("MRREL.RRF"));
		BufferedWriter bw = new BufferedWriter(new FileWriter(
				"formatted_relation_file"));
		String sCurrentLine;
		String[] items;
		long lineNumber = 0;
		while ((sCurrentLine = br.readLine()) != null) {
			if (lineNumber % 1000000 == 0)
				System.out.println(lineNumber);
			if (sCurrentLine.contains("treat")) {
				items = sCurrentLine.split("\\|");
				bw.write(items[0] + " " + items[4]);
				bw.newLine();
			}
			lineNumber++;
		}
		br.close();
		bw.close();
	}

}
