from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import Callable

SYSTEM_PROMPT = """\
<system>
    <role>You are a professional research supporter. You MUST answer ONLY unsing the provided context.</role>
    <language>
        Response with the language what user said.
    </language>
    <mission>
        Answer strictly based on provided academic context.
    </mission>
    <constraints>
        <rule id ="1">All claims must be grounded in the provided context</rule>
        <rule id ="2">Use the retrieved context as the primary source of information.</rule>
        <rule id ="3">If the context provides partial evidence, systhesize it carefully.</rule>
        <rule id ="4">If information is missing, explain what is supported and what remains uncertain.</rule>
        <rule id ="5">Do not refuse unless the context is completely unrelated</rule>
        <rule id ="3">Output must be valid text format</rule>
    </constraints>
</system>
"""

INSUFFICIENT_MSG = "Could not give you an answer since we don't have sufficient documents."

DEFAULT_EMB_MODEL = "BAAI/bge-m3"
DEFAULT_LLM_MODEL = "gpt-4o-mini"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHAT_DB_PATH = str(PROJECT_ROOT / "src/chat_history.db")
DEFAUL_DB_DIR = str(PROJECT_ROOT / "src/chroma_db")
DEFAULT_COLLECTION = "papers"

CHUNCK_SIZE = 900
CHUNK_OVERLAP = 150

TOP_K = 15  # Top k는 5 - 15 정도가 적당. 5는 너무 evidence가 적고, 20은 의미가 흐려지게 됨
MIN_RELEVANCE = 0.2

AUTO_PAPERS_DIR = str(PROJECT_ROOT / "data" / "papers_auto")
ARXIV_MAX_RESULTS = 5
PMC_MAX_RESULTS = 20
PUBMED_MAX_RESULTS = 20


def build_qa_chain(llm_model: str = DEFAULT_LLM_MODEL) -> Callable:
    """Build the QA generation chain used for context-grounded answering.

    Args:
        llm_model (str): OpenAI chat model name.

    Returns:
        Callable: Runnable chain that maps {context, question} to answer text.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "### Context\n{context}\n\n### Question\n{question}\n\n### Answer rule: Base your answer strictly on relevant information from the provided context. If the question and the context are in different languages, translate them as neccessary to locate supporting evidence before answering\n",
            ),
        ]
    )
    llm = ChatOpenAI(model=llm_model, temperature=0)

    return prompt | llm | StrOutputParser()


def build_query_rewrite_chain(llm_model: str = DEFAULT_LLM_MODEL):
    """Build the query rewriting chain for academic web retrieval.

    Args:
        llm_model (str): OpenAI chat model name.

    Returns:
        Callable: Runnable chain that maps {question} to search query text.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You rewrite user questions into a short English academic search query. "
                "Return ONLY the query string, no quotes, no explanations.",
            ),
            (
                "human",
                "User question (may be Korean): {question}\n\n"
                "Rules:\n"
                "- Output in English only\n"
                "- 5~12 keywords max\n"
                "- Prefer biomedical/CS academic terms\n"
                "- If the question contains entities (genes, diseases, methods), keep them verbatim\n"
                "- Avoid full sentences\n"
                "- No punctuation except hyphen\n",
            ),
        ]
    )
    llm = ChatOpenAI(model=llm_model, temperature=0)
    return prompt | llm | StrOutputParser()


QA_CHAIN = build_qa_chain()
QUERY_REWRITE_CHAIN = build_query_rewrite_chain()
