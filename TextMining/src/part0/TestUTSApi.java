package part0;

import gov.nih.nlm.uts.webservice.content.*; 
import gov.nih.nlm.uts.webservice.security.*;
import gov.nih.nlm.uts.webservice.metadata.*;
import gov.nih.nlm.uts.webservice.finder.*;
import gov.nih.nlm.uts.webservice.history.*;
import gov.nih.nlm.uts.webservice.semnet.*;

public class TestUTSApi {

	public static void main(String[] args) {
		try {
			String username = "jajajaqlt";
			String password = "Qts123547";
			UtsWsSecurityController securityService = (new UtsWsSecurityControllerImplService())
					.getUtsWsSecurityControllerImplPort();
			String ticketGrantingTicket = securityService.getProxyGrantTicket(
					username, password);
			String serviceName = "http://umlsks.nlm.nih.gov";
			String singleUseTicket1 = securityService.getProxyTicket(
					ticketGrantingTicket, serviceName);
			String umlsRelease = "2011AB";
			UtsWsContentController utsContentService = (new UtsWsContentControllerImplService())
					.getUtsWsContentControllerImplPort();
			ConceptDTO result1 = utsContentService.getConcept(singleUseTicket1,
					umlsRelease, "C0018787");
			System.out.println(result1.getUi());
			System.out.println(result1.getDefaultPreferredName());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
