from dataclasses import dataclass
import urllib
import json
from config import ARXIV_MAX_RESULTS, PMC_MAX_RESULTS, PUBMED_MAX_RESULTS
import xml.etree.ElementTree as ET


@dataclass
class ArxivPaper:
    title: str
    arxiv_id: str
    summary: str
    pdf_url: str


def arxiv_search(query: str, max_results: int = ARXIV_MAX_RESULTS) -> list[ArxivPaper]:
    """Search arXiv and return paper metadata including PDF URLs.

    Args:
        query (str): Search query string.
        max_results (int): Maximum number of items to retrieve.

    Returns:
        list[ArxivPaper]: Parsed arXiv paper list.
    """

    res = list()

    base = "http://export.arxiv.org/api/query"
    terms = query.split()
    search_query = "+AND+".join(f"all:{urllib.parse.quote(t)}" for t in terms)
    url = f"{base}?search_query={search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"

    with urllib.request.urlopen(url, timeout=20) as resp:
        xml_data = resp.read()

    root = ET.fromstring(xml_data)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("atom:entry", namespace):
        title = (entry.findtext("atom:title", default="", namespaces=namespace) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=namespace) or "").strip()
        id_url = (entry.findtext("atom:id", default="", namespaces=namespace) or "").strip()

        arxiv_id = id_url.rsplit("/", 1)[-1]

        pdf_url = ""
        for link in entry.findall("atom:link", namespace):
            rel = link.attrib.get("rel", "")
            typ = link.attrib.get("type", "")
            href = link.attrib.get("href", "")
            if typ == "application/pdf" or href.endswith(".pdf"):
                pdf_url = href
                break
            if rel == "related" and "pdf" in href:
                pdf_url = href
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if title and arxiv_id and pdf_url:
            res.append(ArxivPaper(title=title, arxiv_id=arxiv_id, summary=summary, pdf_url=pdf_url))

    return res


@dataclass
class PmcPaper:
    title: str
    pmcid: str
    pdf_url: str
    abstract: str


def pmc_efetch_abstract(pmc_ids: list[str]) -> dict[str, str]:
    """Fetch abstracts for PMC articles using the NCBI E-utilities efetch API.

    Args:
        pmc_ids (list[str]): A list of PMC article ids. Each will start with "PMC"
        e.g. PMC123456

    Returns:
        dict[str, str]: A dictionary mapping each PMC_id to its extracted abstract text.
    """

    res = dict()

    if not pmc_ids:
        return res

    url_ids = urllib.parse.quote(",".join(pmc_ids))
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={url_ids}&retmode=xml"
    with urllib.request.urlopen(url, timeout=20) as response:
        xml_text = response.read().decode("utf-8")

    findable_xml = ET.fromstring(xml_text)

    # article nodes
    articles = findable_xml.findall(".//article")

    for idx, pmc_id in enumerate(pmc_ids):
        abstract_text = ""
        if idx < len(articles):
            abstract_nodes = articles[idx].findall(".//abstract")
            if abstract_nodes:
                abstract_text = " ".join("".join(node.itertext()) for node in abstract_nodes).strip()
        res[pmc_id] = abstract_text

    return res


def pmc_search(query: str, max_results: int = PMC_MAX_RESULTS) -> list[PmcPaper]:
    """Search open-access PMC papers and collect downloadable PDF URLs.

    Args:
        query (str): Search query string.
        max_results (int): Maximum number of items to retrieve.

    Returns:
        list[PmcPaper]: PMC paper list with resolved PDF links.
    """

    res = list()

    pmc_ids = list()
    url_query = urllib.parse.quote(query)
    websearch = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pmc&term={url_query}+AND+open+access[filter]&retmode=json&retmax={max_results}"
    )
    with urllib.request.urlopen(websearch, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    id_list = data.get("esearchresult", dict()).get("idlist", list())
    if not id_list:
        print("[PMC] open access 검색 결과가 없습니다.")
        return res

    print(f"[PMC] open access 검색 결과 수: {len(id_list)}")

    ids = ",".join(id_list)
    websummary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi" f"?db=pmc&id={ids}&retmode=json"
    with urllib.request.urlopen(websummary, timeout=20) as response:
        sdata = json.loads(response.read().decode("utf-8"))
    result = sdata.get("result", dict())

    missing_pdf = 0
    for single_id in id_list:
        item = result.get(single_id, dict())
        title = item.get("title", "").strip()

        pmc_id = ""
        for aid in item.get("articleids", []):
            if aid.get("idtype") == "pmcid":
                pmc_id = aid.get("value", "").strip()
                break
        if not pmc_id:
            pmc_id = f"PMC{single_id}"
            pmc_ids.append(pmc_id)

        # OA 서비스로 실제 PDF URL 획득
        pdf_url = _get_oa_pdf_url(pmc_id)
        # abstract 획득
        if title and pdf_url:
            res.append(PmcPaper(title=title, pmcid=pmc_id, pdf_url=pdf_url, abstract=""))
        else:
            missing_pdf += 1

    # add abstract
    abstracts = pmc_efetch_abstract(pmc_ids)
    print(f"[PMC] Successfully get abstract: {len(abstracts)}")
    for pmc_paper in res:
        abstract = abstracts.get(pmc_paper.pmcid, "")
        pmc_paper.abstract = abstract

    if missing_pdf:
        print(f"[PMC] Could not find PDF URL: {missing_pdf}")
    print(f"[PMC] Successfully get PDF URL: {len(res)}")

    return res


@dataclass
class PubmedPaper:
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_date: str


def pubmed_search_abstracts(query: str, max_results: int = PUBMED_MAX_RESULTS) -> list[PubmedPaper]:
    """Search PubMed and fetch article abstracts.

    Args:
        query (str): Search query string.
        max_results (int): Maximum number of items to retrieve.

    Returns:
        list[PubmedPaper]: Papers with title, abstract, and publication metadata.
    """

    res = list()

    url_query = urllib.parse.quote(query)
    websearch = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        f"?db=pubmed&term={url_query}&retmode=json&retmax={max_results}&sort=relevance"
    )
    with urllib.request.urlopen(websearch, timeout=20) as response:
        data = json.loads(response.read().decode("utf-8"))
    id_list = data.get("esearchresult", dict()).get("idlist", list())
    if not id_list:
        print("[PubMed] We could not find any documents")
        return res

    print(f"[PubMed] The number of retrieved documents: {len(id_list)}")

    ids = ",".join(id_list)
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi" f"?db=pubmed&id={ids}&retmode=xml"
    with urllib.request.urlopen(fetch_url, timeout=25) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//PMID") or "").strip()
        title = (article.findtext(".//ArticleTitle") or "").strip()
        journal = (article.findtext(".//Journal/Title") or "").strip()

        year = (article.findtext(".//PubDate/Year") or "").strip()
        medline_date = (article.findtext(".//PubDate/MedlineDate") or "").strip()
        pub_date = year or medline_date

        abstract_texts = list()
        for node in article.findall(".//Abstract/AbstractText"):
            part = "".join(node.itertext()).strip()
            label = (node.attrib.get("Label") or "").strip()
            if part:
                abstract_texts.append(f"{label}: {part}" if label else part)
        abstract = "\n".join(abstract_texts).strip()

        if pmid and title and abstract:
            res.append(PubmedPaper(pmid=pmid, title=title, abstract=abstract, journal=journal, pub_date=pub_date))

    print(f"[PubMed] Successfully extrated abstract: {len(res)}")
    return res


def _get_oa_pdf_url(pmc_id: str) -> str:
    """Resolve an open-access PDF URL from a PMC identifier.

    Args:
        pmc_id (str): PMC ID such as PMC1234567.

    Returns:
        str: HTTPS PDF URL if found; otherwise an empty string.
    """

    oa_api = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmc_id}"
    try:
        with urllib.request.urlopen(oa_api, timeout=10) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        for link in root.iter("link"):
            if link.attrib.get("format") == "pdf":
                href = link.attrib.get("href", "")
                return href.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov")
    except Exception:
        pass
    return ""
