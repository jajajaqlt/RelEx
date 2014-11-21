package part0;

import gov.nih.nlm.nls.lvg.Api.NormApi;

import java.io.File;
import java.util.Scanner;
import java.util.Vector;

public class TestLexicalToolsApi {

	public static void main(String[] args) throws Exception {
		// String lvgConfigFile = "data/config/lvg.properties";
		// LvgCmdApi lvgApi = new LvgCmdApi("-f:i", lvgConfigFile);

		String content = new Scanner(new File("paper.txt")).useDelimiter("\\Z")
				.next();
		System.out.println(content);

		NormApi normalize = new NormApi("data/config/lvg.properties");
		String input2Norm = content;
		Vector<String> outputFromNorm;
		outputFromNorm = normalize.Mutate(input2Norm);
		System.out.println("size is " + outputFromNorm.size());
		for (int i = 0; i < outputFromNorm.size(); i++) {
			System.out.println((String) outputFromNorm.get(i));
			System.out.println(i);
		}
	}

	// public static void main(String[] args) {
	//
	// BufferedReader br = null;
	//
	// try {
	//
	// String sCurrentLine;
	//
	// br = new BufferedReader(
	// new FileReader("data/config/lvg.properties"));
	//
	// while ((sCurrentLine = br.readLine()) != null) {
	// System.out.println(sCurrentLine);
	// }
	//
	// } catch (IOException e) {
	// e.printStackTrace();
	// } finally {
	// try {
	// if (br != null)
	// br.close();
	// } catch (IOException ex) {
	// ex.printStackTrace();
	// }
	// }
	//
	// }
}
