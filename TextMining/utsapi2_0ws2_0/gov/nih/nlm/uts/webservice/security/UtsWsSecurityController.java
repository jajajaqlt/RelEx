
package gov.nih.nlm.uts.webservice.security;

import javax.jws.WebMethod;
import javax.jws.WebParam;
import javax.jws.WebResult;
import javax.jws.WebService;
import javax.xml.ws.RequestWrapper;
import javax.xml.ws.ResponseWrapper;


/**
 * This class was generated by the JAX-WS RI.
 * JAX-WS RI 2.2.4-b01
 * Generated source version: 2.0
 * 
 */
@WebService(name = "UtsWsSecurityController", targetNamespace = "http://webservice.uts.umls.nlm.nih.gov/")
public interface UtsWsSecurityController {


    /**
     * 
     * @param pw
     * @param user
     * @return
     *     returns java.lang.String
     * @throws UtsFault_Exception
     */
    @WebMethod
    @WebResult(targetNamespace = "")
    @RequestWrapper(localName = "getProxyGrantTicket", targetNamespace = "http://webservice.uts.umls.nlm.nih.gov/", className = "gov.nih.nlm.uts.webservice.security.GetProxyGrantTicket")
    @ResponseWrapper(localName = "getProxyGrantTicketResponse", targetNamespace = "http://webservice.uts.umls.nlm.nih.gov/", className = "gov.nih.nlm.uts.webservice.security.GetProxyGrantTicketResponse")
    public String getProxyGrantTicket(
        @WebParam(name = "user", targetNamespace = "")
        String user,
        @WebParam(name = "pw", targetNamespace = "")
        String pw)
        throws UtsFault_Exception
    ;

    /**
     * 
     * @param service
     * @param tgt
     * @return
     *     returns java.lang.String
     * @throws UtsFault_Exception
     */
    @WebMethod
    @WebResult(targetNamespace = "")
    @RequestWrapper(localName = "getProxyTicket", targetNamespace = "http://webservice.uts.umls.nlm.nih.gov/", className = "gov.nih.nlm.uts.webservice.security.GetProxyTicket")
    @ResponseWrapper(localName = "getProxyTicketResponse", targetNamespace = "http://webservice.uts.umls.nlm.nih.gov/", className = "gov.nih.nlm.uts.webservice.security.GetProxyTicketResponse")
    public String getProxyTicket(
        @WebParam(name = "TGT", targetNamespace = "")
        String tgt,
        @WebParam(name = "service", targetNamespace = "")
        String service)
        throws UtsFault_Exception
    ;

    /**
     * 
     * @param ticket
     * @param service
     * @return
     *     returns java.lang.String
     * @throws UtsFault_Exception
     */
    @WebMethod
    @WebResult(targetNamespace = "")
    @RequestWrapper(localName = "validateProxyTicket", targetNamespace = "http://webservice.uts.umls.nlm.nih.gov/", className = "gov.nih.nlm.uts.webservice.security.ValidateProxyTicket")
    @ResponseWrapper(localName = "validateProxyTicketResponse", targetNamespace = "http://webservice.uts.umls.nlm.nih.gov/", className = "gov.nih.nlm.uts.webservice.security.ValidateProxyTicketResponse")
    public String validateProxyTicket(
        @WebParam(name = "ticket", targetNamespace = "")
        String ticket,
        @WebParam(name = "service", targetNamespace = "")
        String service)
        throws UtsFault_Exception
    ;

}