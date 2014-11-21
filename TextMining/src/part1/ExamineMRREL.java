package part1;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeSet;

public class ExamineMRREL {

	public static void main(String[] args) {

		BufferedReader br = null;
		MRRELRecord rec = null;
		ArrayList<MRRELRecord> recs = new ArrayList<MRRELRecord>();
		Set<String> relsBwSameConcepts = new HashSet<String>();
		Set<String> addRelsBwSameConcepts = new HashSet<String>();
		// Set<String> CUI1s = new HashSet<String>();
		// Set<String> CUI2s = new HashSet<String>();
		Set<String> diffCUI1s = new TreeSet<String>();
		Set<String> diffCUI2s = new TreeSet<String>();
		int index = 0;

		String CUI1, AUI1, REL, RELA, CUI2, AUI2, oldCUI1 = null, oldCUI2 = null;
		try {
			String sCurrentLine;
			br = new BufferedReader(new FileReader("MRREL.RRF"));
			while ((sCurrentLine = br.readLine()) != null) {
				rec = new MRRELRecord();
				rec.index = index;
				rec.items = sCurrentLine.split("\\|", 17);
				CUI1 = rec.getCUI1();
				CUI2 = rec.getCUI2();
				if (CUI1.equals(CUI2)) {
					REL = rec.getREL();
					RELA = rec.getRELA();
					relsBwSameConcepts.add(REL);
					addRelsBwSameConcepts.add(RELA);
				} else {
					if (CUI1.equals(oldCUI1) && CUI2.equals(oldCUI2)) {
						diffCUI1s.add(CUI1);
						diffCUI2s.add(CUI2);
						recs.add(rec);
					}
				}
				oldCUI1 = CUI1;
				oldCUI2 = CUI2;
				index++;
			}
			System.out.println("Relations b/w same concepts are: "
					+ relsBwSameConcepts.toString());
			System.out.println("Additional relations b/w same concepts are: "
					+ addRelsBwSameConcepts.toString());
			System.out.println("CUI1s are: " + diffCUI1s.toString());
			System.out.println("CUI2s are: " + diffCUI2s.toString());
			System.out.println("Size of recs is: " + recs.size());
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
