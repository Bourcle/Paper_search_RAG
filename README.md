# PDF RAG Chatbot (with Chat History)

업로드한 PDF와 자동 웹 검색(PMC/PubMed/arXiv) 기반으로 답변하는 **연구 보조형 RAG 챗봇**입니다.  
Gradio UI, Chroma 벡터DB, SQLite 세션 히스토리를 사용합니다.

## 1. 주요 기능

- PDF 업로드 후 자동 청킹 및 벡터 인덱싱
- 세션별 대화 이력 저장/불러오기/삭제
- 메타데이터 필터 기반 질의 (`@file`, `@page`, `@doc_id`, `@filter`)
- 검색: Chroma dense + local BM25 sparse 결합 (서버리스 하이브리드)
  - Stage1에서 dense/sparse를 병렬 조회 후 점수 결합
- 근거 부족 시 웹 자동 보강
  - 1순위: PMC Open Access PDF
  - 2순위: PubMed Abstract
  - 3순위: arXiv PDF
- 스트리밍 응답 UI

## 2. 기술 스택

- LLM: `langchain-openai` (`ChatOpenAI`)
- Embedding: `BAAI/bge-m3` (`HuggingFaceEmbeddings`)
- Vector DB: `Chroma`
- UI: `Gradio`
- Session DB: `SQLite`
- PDF 처리: `PyPDFLoader`, `RecursiveCharacterTextSplitter`

## 3. 프로젝트 구조

```text
.
├─ src/
│  ├─ main.py                    # 앱 실행 엔트리포인트
│  ├─ config.py                  # 전역 설정/프롬프트/체인 초기화
│  ├─ chat_history.db            # SQLite 대화 이력(DB 실행 후 생성)
│  ├─ chroma_db/                 # Chroma 영속 저장소(DB 실행 후 생성)
│  ├─ app/
│  │  ├─ app_process.py          # Gradio 컴포넌트 구성 및 이벤트 바인딩
│  │  └─ ui_utils.py             # UI 액션 처리(전송/업로드/세션 핸들링)
│  ├─ database/
│  │  ├─ sessions.py             # SQLite 세션/메시지 CRUD
│  │  └─ vector_db.py            # Chroma 열기, 검색/응답, 자동 수집/적재
│  ├─ retriever/
│  │  ├─ pdf_utils.py            # PDF 검증/로딩/청킹/다운로드
│  │  ├─ db_retriever.py         # Dense + Sparse 병렬 하이브리드 검색 및 context 포맷
│  │  ├─ sparse_index.py         # SQLite FTS5 기반 증분 sparse 인덱스
│  │  └─ web_retriever.py        # arXiv/PMC/PubMed 검색
│  └─ utils/
│     └─ utils.py                # 질의 필터 파싱, 한국어 감지
├─ data/                         # 입력/자동 수집 문서 저장 디렉토리
└─ README.md
```

## 4. 동작 흐름

1. 사용자가 질문을 보냄
2. `answer_from_db()`에서 dense/sparse를 Stage1 병렬 조회 후 점수 결합
3. 관련 문서 점수가 충분하면 QA 체인으로 답변 생성
4. 근거 부족(`INSUFFICIENT_MSG`)이면 `auto_fetch_and_ingest()` 실행
5. 웹 문서 인입 후 같은 질문으로 1회 재시도
6. 결과를 스트리밍으로 UI에 표시하고 세션 DB에 저장

## 5. 사전 준비

- Python `3.12+`
- OpenAI API Key
- 인터넷 연결(웹 검색/논문 다운로드 시)

`.env` 파일 예시:

```bash
OPENAI_API_KEY=your_openai_api_key
```

## 6. 설치

### uv 사용 (권장)

```bash
uv sync
```

### pip 사용

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 7. 실행 방법

프로젝트 루트에서:

```bash
uv run python src/main.py
```

또는(venv 활성화 상태):

```bash
python3 src/main.py
```

실행 후 Gradio 로컬 주소(예: `http://127.0.0.1:7860`)로 접속합니다.

## 8. 사용 방법

1. 좌측 `Upload PDF`에서 PDF 업로드
2. 우측 질문창에 질문 입력 후 `Send`
3. 필요 시 세션 필터 JSON 입력
4. 대화 이력은 좌측 세션 드롭다운에서 전환

### 인라인 필터 예시

- 파일 기준: `Summarize key points @file=paper.pdf`
- 페이지 기준: `What is the experiment setup? @file=paper.pdf @page=3`
- 문서 ID 기준: `Explain this section @doc_id=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`
- JSON 필터: `Give conclusions @filter={"filename":{"$eq":"paper"}}`

### 세션 필터(JSON) 예시

```json
{"filename": {"$eq": "paper"}}
```

인라인 필터와 세션 필터가 동시에 있으면 `$and`로 병합됩니다.

## 9. 주요 설정값 (`src/config.py`)

- `DEFAULT_EMB_MODEL`: 임베딩 모델 (`BAAI/bge-m3`)
- `DEFAULT_LLM_MODEL`: 답변 생성 모델 (`gpt-4o-mini`)
- `TOP_K`: 검색 문서 수
- `DENSE_K_MULTIPLIER`: dense 후보 배수 (`dense_k = TOP_K * multiplier`)
- `W_DENSE`, `W_SPARSE`: 하이브리드 점수 가중치
- `MIN_RELEVANCE`: 최소 관련도 임계값
- `CHUNCK_SIZE`, `CHUNK_OVERLAP`: PDF 청킹 파라미터
- `AUTO_PAPERS_DIR`: 웹 자동 수집 PDF 저장 경로
- `ARXIV_MAX_RESULTS`, `PMC_MAX_RESULTS`, `PUBMED_MAX_RESULTS`: 소스별 검색 수

## 10. 데이터 저장 위치

- 세션/메시지: `./chat_history.db`
- 벡터 임베딩/메타데이터: `./chroma_db`
- 자동 다운로드 논문: `./data/papers_auto`

## 11. 트러블슈팅

- `OPENAI_API_KEY` 누락: `.env` 또는 환경변수 설정 확인
- PDF 업로드 실패: 파일 확장자/손상 여부 확인 (`.pdf`만 지원)
- 웹 보강 실패: 네트워크 상태, 원문 링크 접근 가능 여부 확인
- 응답 근거 부족: 더 관련성 높은 PDF 업로드 또는 필터 완화

## 12. 개선 포인트

- 로깅 체계 표준화(`print` -> `logging`)
- 예외 타입 세분화 및 사용자 메시지 정교화
- 테스트 코드(단위/통합) 추가
