from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import psycopg2
import psycopg2.extras

# ── RAG imports ──────────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

# ── Configurações ─────────────────────────────────────────
DATABASE_URL = "postgresql://neondb_owner:npg_w1a5UZthFdEl@ep-bitter-star-agw2p5s2-pooler.c-2.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

CHROMA_PATH = r"C:\Users\Alexandr\Desktop\Universidade\3º Ano\Projeto\Testes ChatBot\chatbot_rag\chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# ── Ligação Neon ─────────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# ── App ──────────────────────────────────────────────────
app = FastAPI(title="OP Report API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://joseruisilva.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Schemas ───────────────────────────────────────────────
class ReportIn(BaseModel):
    source_code: str
    file_name: str
    report_url: str
    publication_date: datetime

class ChatIn(BaseModel):
    question: str

class OpDataIn(BaseModel):
    report_id: int
    file_name: str
    file_url: str
    extract_function: str
    file_type: str

# ── Helpers RAG ───────────────────────────────────────────
def get_embedding_function():
    return OllamaEmbeddings(model="mxbai-embed-large")

def query_rag(query_text: str):
    embedding_function = get_embedding_function()

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
    )

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    return {"answer": response_text, "sources": sources}


# ── Endpoints INSERÇÃO (POST) ─────────────────────────────

@app.post("/op_report", status_code=201)
def add_report(report: ReportIn):
    """Insere um novo relatório na tabela op_report."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO op_report (source_code, file_name, report_url, publication_date)
            VALUES (%s, %s, %s, %s)
            RETURNING report_id;
        """, (report.source_code, report.file_name, report.report_url, report.publication_date))
        report_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return {"report_id": report_id, "message": "Relatório inserido com sucesso."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/op_data", status_code=201)
def add_op_data(data: OpDataIn):
    """Insere um novo registo na tabela op_data, verificando a existência do report_id."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT 1 FROM op_report WHERE report_id = %s", (data.report_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail=f"Erro: O report_id {data.report_id} não existe na base de dados.")

        cur.execute("""
            INSERT INTO op_data (report_id, file_name, file_url, extract_function, file_type)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING file_id;
        """, (data.report_id, data.file_name, data.file_url, data.extract_function, data.file_type))
        
        file_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return {"file_id": file_id, "message": "Ficheiro op_data inserido com sucesso."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Endpoints LEITURA (GET) ─────────────────────────────

@app.get("/op_report")
def get_reports():
    """Devolve todos os registos da tabela op_report."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT report_id, source_code, file_name, report_url, publication_date FROM op_report ORDER BY report_id DESC;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources")
def get_sources():
    """Devolve todas as fontes da tabela dim_source."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT source_code, source_name FROM dim_source ORDER BY source_name;")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indicators")
def get_indicators(source_code: str = None):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        if source_code:
            cur.execute("""
                SELECT DISTINCT i.indicator_code, i.indicator_name, r.source_code
                FROM dim_indicator i
                JOIN fact_values f ON f.indicator_code = i.indicator_code
                JOIN op_report r ON r.report_id = f.report_id
                WHERE r.source_code = %s
                ORDER BY i.indicator_name;
            """, (source_code,))
        else:
            cur.execute("""
                SELECT DISTINCT ON (i.indicator_code)
                    i.indicator_code, i.indicator_name, r.source_code
                FROM dim_indicator i
                JOIN fact_values f ON f.indicator_code = i.indicator_code
                JOIN op_report r ON r.report_id = f.report_id
                ORDER BY i.indicator_code;
            """)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fact_values")
def get_fact_values(indicator_code: str):
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = """
            SELECT 
                c.location_code AS location_code, 
                c.location_name AS location_name,
                d.year AS year,
                f.value AS value
            FROM fact_values f
            JOIN dim_location c ON f.location_code = c.location_code
            JOIN dim_indicator i ON f.indicator_code = i.indicator_code
            JOIN dim_date d ON f.date_id = d.date_id
            WHERE i.indicator_code = %s
            ORDER BY d.year ASC, c.location_name ASC;
        """
        cur.execute(query, (indicator_code,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Endpoint CHAT ─────────────────────────────────────────

@app.post("/chat")
def chat(body: ChatIn):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")
    try:
        result = query_rag(body.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))