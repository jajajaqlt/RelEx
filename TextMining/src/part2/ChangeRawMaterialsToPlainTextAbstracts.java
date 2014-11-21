package part2;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;

public class ChangeRawMaterialsToPlainTextAbstracts {

	public static void main(String[] args) throws Exception {
		String inputFilename = "ohsumed.88-91";
		StringBuffer sb = new StringBuffer();
		BufferedReader br = new BufferedReader(new FileReader(inputFilename));
		String line;
		int count = 0;
		boolean flag = false;
		while ((line = br.readLine()) != null) {
			if (!flag && line.equals(".W"))
				flag = true;
			else if (flag) {
				sb.append(line).append("\n");
				flag = false;
				count++;
			} else {
				// other info.
			}
		}
		br.close();
		String input = sb.toString();
		System.out.println(count);
		BufferedWriter bw = new BufferedWriter(new FileWriter(
				"abstracts_plain_text"));
		bw.write(input);
		bw.close();
	}

}
