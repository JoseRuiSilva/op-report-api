from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import psycopg2
import psycopg2.extras
import os

# ── SQL Agent imports ─────────────────────────────────────
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq

# ── Configurações ─────────────────────────────────────────
DATABASE_URL = os.environ["DATABASE_URL"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# ── Ligação Neon ─────────────────────────────────────────
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# ── SQL Agent (inicializado uma vez no arranque) ──────────
SYSTEM_PROMPT = """
És um assistente perito em Ciência de Dados e PostgreSQL.
A base de dados funciona num Modelo Dimensional (Star Schema).
A tabela central é a `fact_values` (colunas: report_id, location_code, indicator_code, date_id, value, value_type).
Para obteres nomes legíveis, tens OBRIGATORIAMENTE de fazer JOIN com as tabelas de dimensão:
- JOIN dim_location l ON fact_values.location_code = l.location_code (para obteres l.location_name)
- JOIN dim_indicator i ON fact_values.indicator_code = i.indicator_code (para obteres i.indicator_name)
- JOIN dim_date d ON fact_values.date_id = d.date_id (para obteres d.year)

PASSO 1: PADRÃO OBRIGATÓRIO PARA RANKINGS E POSIÇÕES
Se o utilizador pedir uma posição, ranking, "melhor lugar", etc., É PROIBIDO filtrar pelo país na query principal.
Tens OBRIGATORIAMENTE de usar este modelo exato (com subquery, PARTITION BY year e JOINs):

SELECT year, ranking, value FROM (
    SELECT
        l.location_name as country,
        d.year,
        f.value,
        RANK() OVER (PARTITION BY d.year ORDER BY f.value DESC) as ranking
    FROM fact_values f
    JOIN dim_location l ON f.location_code = l.location_code
    JOIN dim_date d ON f.date_id = d.date_id
    WHERE f.indicator_code = 'AQUI_A_SIGLA_DO_INDICADOR_EX_BCA'
) subquery
WHERE country = 'AQUI_O_NOME_DO_PAIS'
ORDER BY ranking ASC LIMIT 1;

PASSO 2: EXECUÇÃO OBRIGATÓRIA
NÃO te limites a escrever o SQL corrigido no texto.
Tens OBRIGATORIAMENTE de usar a ferramenta `sql_db_query` para executar o código e obter os números finais reais.

PASSO 3: FORMATO DA RESPOSTA
1. Responde à pergunta com os números reais devolvidos.
2. Inclui a query SQL executada num bloco markdown (```sql ... ```).
"""

def build_agent():
    db = SQLDatabase.from_uri(DATABASE_URL)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=GROQ_API_KEY
    )
    agent = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="tool-calling",
        verbose=False,
        prefix=SYSTEM_PROMPT,
        return_intermediate_steps=False
    )
    return agent

# Inicializa o agente uma vez (lazy, na primeira chamada)
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent

# ── App ──────────────────────────────────────────────────
app = FastAPI(title="OP Report API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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
    """Responde a perguntas em linguagem natural usando um SQL Agent com Groq + LLaMA."""
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")
    try:
        agent = get_agent()
        result = agent.invoke({"input": body.question})
        answer = result.get("output", str(result))
        return {"answer": answer, "sources": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))