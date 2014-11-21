package part1;

import gov.nih.nlm.nls.metamap.Ev;  
import gov.nih.nlm.nls.metamap.Mapping;
import gov.nih.nlm.nls.metamap.MetaMapApi;
import gov.nih.nlm.nls.metamap.MetaMapApiImpl;
import gov.nih.nlm.nls.metamap.PCM;
import gov.nih.nlm.nls.metamap.Result;
import gov.nih.nlm.nls.metamap.Utterance;
import gov.nih.nlm.uts.webservice.content.ConceptRelationDTO;
import gov.nih.nlm.uts.webservice.content.UtsWsContentController;
import gov.nih.nlm.uts.webservice.content.UtsWsContentControllerImplService;
import gov.nih.nlm.uts.webservice.security.UtsWsSecurityController;
import gov.nih.nlm.uts.webservice.security.UtsWsSecurityControllerImplService;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class GetRels {

	public static void main(String[] args) throws Exception {
		getMRRELRecords("MRREL.RRF");

		String inputFilename = args[0];
		// String outputFilename = args[1];

		// reads in file and makes it into a string
		StringBuffer sb = new StringBuffer();
		BufferedReader br = new BufferedReader(new FileReader(inputFilename));
		String line;
		while ((line = br.readLine()) != null) {
			sb.append(line).append("\n");
		}
		br.close();
		String input = sb.toString();
		System.out.println("input: " + input);

		ArrayList<part1.Utterance> utterances = new ArrayList<part1.Utterance>();
		MetaMapApi api = new MetaMapApiImpl(0);
		List<Result> resultList = api.processCitationsFromString(input);
		// should be only one result because of only one document
		for (Result result : resultList) {
			for (Utterance utterance : result.getUtteranceList()) {
				part1.Utterance utt = new part1.Utterance();

				for (PCM pcm : utterance.getPCMList()) {
					utt.phrases.add(pcm.getPhrase().getPhraseText());
					if (pcm.getMappingList().size() == 0) {
						utt.phrases.remove(utt.phrases.size() - 1);
					}
					for (Mapping map : pcm.getMappingList()) {
						for (Ev mapEv : map.getEvList()) {
							utt.CUIs.add(mapEv.getConceptId());
							utt.preferredNames.add(mapEv.getPreferredName());
							utt.conceptNames.add(mapEv.getConceptName());
							break;
						}
						break;
					}
					// end of mapping
				}
				// end of foreach phrase
				utterances.add(utt);
			}
			// end of foreach utterance
		}

		// for (part1.Utterance utt : utterances) {
		// System.out.println("New Utterance:");
		// for (int i = 0; i < utt.phrases.size(); i++) {
		// System.out.println("Phrase: " + utt.phrases.get(i));
		// System.out.println("CUI:" + utt.CUIs.get(i));
		// System.out.println("Concept Name: " + utt.conceptNames.get(i));
		// System.out.println("Preferred Name: " + utt.preferredNames.get(i));
		// System.out.println();
		// }
		// }
		String username = "jajajaqlt";
		String password = "Qts123547";
		String umlsRelease = "2011AB";
		String serviceName = "http://umlsks.nlm.nih.gov";

		UtsWsContentController utsContentService = (new UtsWsContentControllerImplService())
				.getUtsWsContentControllerImplPort();
		UtsWsSecurityController securityService = (new UtsWsSecurityControllerImplService())
				.getUtsWsSecurityControllerImplPort();
		String ticketGrantingTicket = securityService.getProxyGrantTicket(
				username, password);
		gov.nih.nlm.uts.webservice.content.Psf myPsf = new gov.nih.nlm.uts.webservice.content.Psf();
		List<ConceptRelationDTO> myConceptRelationsDTO;
		String singleUseTicket;
		ConceptRelationDTO myConceptRelationDTO;

		ArrayList<String> CUIs;
		ArrayList<String> preferredNames;
		String CUI;
		String preferredName;
		String otherConceptRela;
		String otherConceptUi;
		String otherConceptName;

		for (part1.Utterance utt : utterances) {
			System.out.println("New Utterance");
			CUIs = utt.CUIs;
			preferredNames = utt.preferredNames;

			for (int i = 0; i < CUIs.size(); i++) {
				CUI = CUIs.get(i);
				preferredName = preferredNames.get(i);
				singleUseTicket = securityService.getProxyTicket(
						ticketGrantingTicket, serviceName);
				myConceptRelationsDTO = new ArrayList<ConceptRelationDTO>();
				myConceptRelationsDTO = utsContentService
						.getConceptConceptRelations(singleUseTicket,
								umlsRelease, CUI, myPsf);
				for (int j = 0; j < myConceptRelationsDTO.size(); j++) {
					myConceptRelationDTO = myConceptRelationsDTO.get(j);
					otherConceptUi = myConceptRelationDTO.getRelatedConcept()
							.getUi();
					otherConceptRela = myConceptRelationDTO
							.getAdditionalRelationLabel();
					String otherConceptRel = myConceptRelationDTO
							.getRelationLabel();
					// String otherConceptRela =
					// myConceptRelationDTO.getAdditionalRelationLabel();

					otherConceptName = myConceptRelationDTO.getRelatedConcept()
							.getDefaultPreferredName();
					if (utt.CUIs.contains(otherConceptUi)) {
						System.out.println("Found relationship: "
								+ preferredName + " : " + otherConceptRel
								+ " : " + otherConceptName);
						// System.out.println();
					}
				}

			}
		}

	}

	public static void getMRRELRecords(String fileName) {
		ArrayList<MRRELRecord> recs = new ArrayList<MRRELRecord>();
		BufferedReader br = null;
		MRRELRecord rec = null;
		int index = 0;
		try {
			String sCurrentLine;
			br = new BufferedReader(new FileReader(fileName));
			while ((sCurrentLine = br.readLine()) != null) {
				rec = new MRRELRecord();
				rec.index = index;
				index++;
				rec.items = sCurrentLine.split("\\|", 17);
				recs.add(rec);
			}
			System.out.println("There are " + index + " records in total");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}