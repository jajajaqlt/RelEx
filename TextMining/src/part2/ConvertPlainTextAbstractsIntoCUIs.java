package part2;

import gov.nih.nlm.nls.metamap.Ev;
import gov.nih.nlm.nls.metamap.MetaMapApi;
import gov.nih.nlm.nls.metamap.MetaMapApiImpl;
import gov.nih.nlm.nls.metamap.PCM;
import gov.nih.nlm.nls.metamap.Result;
import gov.nih.nlm.nls.metamap.Utterance;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * Description: inputs an abstract, outpus a line of CUIs and indices of CUIs of type "dsyn", "mobd" and "neop"
 * @author letaoqi
 *
 */
public class ConvertPlainTextAbstractsIntoCUIs {

	public static void main(String[] args) throws Exception {
		Date date = new Date();
		System.out.println("Start processing at: " + date.toString());

		MetaMapApi api = new MetaMapApiImpl(0);
		List<Result> resultList;
		// BufferedReader br = new BufferedReader(new FileReader(
		// "abstracts_plain_text"));
		BufferedReader br = new BufferedReader(new FileReader("abstracts_test"));

		BufferedWriter bw = new BufferedWriter(new FileWriter(
				"formatted_output_test"));
		StringBuffer sb = new StringBuffer();

		String line, outputIndexLine, conceptId, STCode, outputCUILineSpaceSeparated = "";
		ArrayList<String> outputCUILine = new ArrayList<String>();

		Set<String> CUISet = new HashSet<String>();

		int index = 1;

		while ((line = br.readLine()) != null) {
			System.out.println("Processing #" + index + " abstract.");
			outputCUILine.clear();
			outputIndexLine = "";
			outputCUILineSpaceSeparated = "";

			resultList = api.processCitationsFromString(line);
			for (Result result : resultList) {
				for (Utterance utterance : result.getUtteranceList()) {
					for (PCM pcm : utterance.getPCMList()) {
						if (pcm.getMappingList().size() != 0) {
							for (Ev mapEv : pcm.getMappingList().get(0)
									.getEvList()) {
								conceptId = mapEv.getConceptId();
								outputCUILine.add(conceptId);
								outputCUILineSpaceSeparated += conceptId + " ";
								CUISet.add(conceptId);
								STCode = mapEv.getSemanticTypes().get(0);
								if (STCode.equals("dsyn")
										|| STCode.equals("mobd")
										|| STCode.equals("neop")) {
									outputIndexLine += outputCUILine.size() - 1
											+ " ";
								}
							}
						}
					}
				}
			}
			sb.append(outputCUILineSpaceSeparated).append("\n");
			sb.append(outputIndexLine).append("\n");
			index++;
		}

		StringBuffer sbh = new StringBuffer();
		sbh.append("" + CUISet.size() + "\n");
		Iterator<String> iter = CUISet.iterator();
		while (iter.hasNext()) {
			sbh.append(iter.next() + " ");
		}
		sbh.append("\n");
		sbh.append(sb);
		br.close();
		bw.write(sbh.toString());
		bw.close();

		date = new Date();
		System.out.println("Closing program at: " + date.toString());
	}
}
