"""
Credit Risk Query Engine — Streamlit App
----------------------------------------------------------
Takes a natural-language commercial lending portfolio question,
routes it through the query engine pipeline, validates the SQL,
executes it against the local SQLite database, and generates a
concise business answer.

Run with:
    streamlit run app.py

Expects, in the same folder as this file:
    - credit_risk_portfolio.db
    - OPENAI_API_KEY and OPENAI_API_BASE in Streamlit secrets
"""

import json
import os
import re
import sqlite3
import warnings

import pandas as pd
import sqlparse
import streamlit as st

from langchain_openai import ChatOpenAI


warnings.filterwarnings("ignore")


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# Configuration / LLM / Database Setup
# ============================================================

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_API_BASE = st.secrets["OPENAI_API_BASE"]

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE


@st.cache_resource
def get_llms():
    """
    Creates the two LLMs used by the query engine.

    Primary LLM:
    - Intent classification
    - SQL generation
    - SQL retry
    - Response generation

    Evaluator LLM:
    - SQL relevance validation
    """

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    return llm, evaluator_llm


@st.cache_resource
def get_db_connection():
    """
    Opens the SQLite database in read-only mode.
    """

    db_path = "credit_risk_portfolio.db"

    return sqlite3.connect(
        f"file:{db_path}?mode=ro",
        uri=True,
        check_same_thread=False
    )


llm, evaluator_llm = get_llms()
conn = get_db_connection()


# ============================================================
# Database Schema
# ============================================================

DATABASE_SCHEMA = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# ============================================================
# Verified Query Template Library
# ============================================================

VERIFIED_QUERY_LIBRARY = {

    "VQ1": {
        "description":
            "Sector-wise total outstanding and NPA amount breakdown "
            "across all sectors",

        "sql": """
SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_million,
    ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss') THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_outstanding_million
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_million DESC
"""
    },

    "VQ2": {
        "description":
            "Total portfolio outstanding broken down by loan category "
            "(Corporate, Mid-Corporate, SME)",

        "sql": """
SELECT
    loan_category,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_million,
    COUNT(loan_account_number) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_million DESC
"""
    },

    "VQ3": {
        "description":
            "IFRS 9 stage-wise summary showing loan count, exposure at "
            "default, and expected credit loss for the latest quarter",

        "sql": """
SELECT
    ifrs9_stage,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_million,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_million
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
"""
    },

    "VQ4": {
        "description":
            "Average provision coverage ratio by sector for the latest "
            "reporting quarter",

        "sql": """
SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC
"""
    },

    "VQ5": {
        "description":
            "Top 10 largest loan exposures by outstanding amount at the "
            "borrower level",

        "sql": """
SELECT
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_million,
    l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10
"""
    },

    "VQ6": {
        "description":
            "Top 5 largest exposures aggregated at the business group level",

        "sql": """
SELECT
    group_name,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_million
FROM loan_master
WHERE group_name IS NOT NULL AND group_name != ''
GROUP BY group_name
ORDER BY total_outstanding_million DESC
LIMIT 5
"""
    },

    "VQ7": {
        "description":
            "All overdue loan accounts with their days past due and "
            "asset classification",

        "sql": """
SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_million,
    l.days_past_due,
    l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC
"""
    },

    "VQ8": {
        "description":
            "Distribution of loans across days-past-due buckets showing "
            "aging profile of the portfolio",

        "sql": """
SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        WHEN days_past_due > 90 THEN '90+'
    END AS dpd_bucket,
    COUNT(loan_account_number) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_million
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        WHEN '90+' THEN 5
    END ASC
"""
    },

    "VQ9": {
        "description":
            "Borrowers whose internal rating was downgraded in the latest "
            "rating cycle",

        "sql": """
SELECT
    borrower_id,
    previous_rating,
    internal_rating AS current_internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC
"""
    },

    "VQ10": {
        "description":
            "Expected credit loss trend across all reporting quarters "
            "showing provisioning movement over time",

        "sql": """
SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_million
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC
"""
    }
}


# ============================================================
# Tool 1: Intent Classification
# ============================================================

def classify_intent(user_question, query_library):

    library_descriptions = "\n".join(
        [
            f"{qid}: {entry['description']}"
            for qid, entry in query_library.items()
        ]
    )

    classification_prompt = f"""
### ROLE

You are a query router for a commercial lending credit-risk analytics
system at Northbridge Bank.

Your job is to decide whether a business user's question can be
answered by one of the pre-approved query templates or whether
fresh SQL generation is required.

### USER QUESTION

{user_question}

### AVAILABLE VERIFIED QUERY TEMPLATES

{library_descriptions}

### INSTRUCTIONS

1. Identify the analytical intent of the question.
2. Compare the intent against every verified template.
3. Match based on semantic meaning, not exact wording.
4. "Non-performing" or "NPA" means
   asset_classification IN ('Substandard', 'Doubtful', 'Loss').
5. "Overdue" or "past due" means days_past_due > 0.
6. "Restructured" refers to is_restructured = 1.
7. "Impaired" means IFRS 9 Stage 3.
8. "Book" or "exposure" means total_outstanding.
9. Only select a verified template if it genuinely answers
   the requested question.
10. A row-level question (e.g., "list every restructured loan")
    should not match an aggregate template.
11. A trend question (e.g., "over the last four quarters") should
    not match a template that returns a single reporting date.
12. If no template genuinely answers the question, use the
    generated route.

### OUTPUT

Return ONLY valid JSON:

{{
    "route": "verified" or "generated",
    "query_id": "VQ1" ... "VQ10" or null,
    "match_reason": "one short sentence"
}}

Do not include any other text.
"""

    response = llm.invoke(
        classification_prompt
    ).content.strip()

    json_match = re.search(
        r"\{.*\}",
        response,
        re.DOTALL
    )

    if json_match:

        try:
            return json.loads(json_match.group())

        except json.JSONDecodeError:
            pass

    return {
        "route": "generated",
        "query_id": None,
        "match_reason": "Could not parse classification"
    }


# ============================================================
# Tool 2: Query Generation
# ============================================================

def generate_query(user_question, schema_context):

    generation_prompt = f"""
### ROLE

You are a senior SQL developer specializing in commercial lending
credit-risk analytics using SQLite.

### USER QUESTION

{user_question}

### DATABASE SCHEMA

{schema_context}

### INSTRUCTIONS

1. Write a single SQL query that answers the user's question.
2. Use only the provided schema.
3. The query must be read-only.
4. Use SELECT or WITH ... SELECT only.
5. Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
6. Do not invent tables or columns.
7. Resolve named sectors using sector_name or sector_code
   (e.g., 'Real Estate' maps to SEC_RE, 'Hospitality' maps to SEC_HOSP).
8. Use SQLite-compatible syntax.
9. Never subtract dates directly; use julianday(a) - julianday(b)
   for day differences.
10. Alias numeric columns with meaningful unit suffixes:
    - _usd
    - _pct
    - _count
    - _days
11. provisioning has one row per loan per reporting_date. Filter to
    the latest reporting_date for current figures, or GROUP BY
    reporting_date for trends.

### OUTPUT

Return ONLY the SQL query.

No markdown.
No comments.
No explanation.
"""

    sql = llm.invoke(
        generation_prompt
    ).content.strip()

    sql = re.sub(
        r"^```sql\s*|\s*```$",
        "",
        sql,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()

    sql = re.sub(
        r"^```\s*|\s*```$",
        "",
        sql,
        flags=re.MULTILINE
    ).strip()

    return sql


# ============================================================
# Tool 3: Query Validation
# ============================================================

def validate_query(
    user_question,
    candidate_sql,
    db_connection,
    query_library,
    query_id=None
):

    result = {
        "passed": False,
        "failed_check": None,
        "details": "",
        "relevance_confidence": None
    }

    # --------------------------------------------------------
    # Check 1: Read-only shape
    # --------------------------------------------------------

    sql_upper = candidate_sql.upper().strip()

    forbidden_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "REPLACE",
        "ATTACH"
    ]

    if not (
        sql_upper.startswith("SELECT")
        or sql_upper.startswith("WITH")
    ):
        result["failed_check"] = "read_only_shape"
        result["details"] = "Query must start with SELECT or WITH"
        return result

    for kw in forbidden_keywords:

        if re.search(
            r"\b" + kw + r"\b",
            sql_upper
        ):
            result["failed_check"] = "read_only_shape"
            result["details"] = (
                f"Forbidden keyword detected: {kw}"
            )
            return result

    if ";" in candidate_sql.rstrip().rstrip(";").rstrip():

        result["failed_check"] = "read_only_shape"
        result["details"] = (
            "Multiple statements are not allowed"
        )
        return result

    # --------------------------------------------------------
    # Check 2: Schema conformance
    # --------------------------------------------------------

    cur = db_connection.cursor()

    real_tables = [
        r[0]
        for r in cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """
        ).fetchall()
    ]

    real_columns = set()

    for table in real_tables:

        for col_info in cur.execute(
            f"PRAGMA table_info({table})"
        ).fetchall():

            real_columns.add(
                col_info[1].lower()
            )

    parsed = sqlparse.parse(candidate_sql)[0]

    referenced_identifiers = re.findall(
        r"\b[a-z_][a-z0-9_]*\b",
        candidate_sql.lower()
    )

    sql_keywords = {
        "select",
        "from",
        "where",
        "and",
        "or",
        "group",
        "by",
        "order",
        "having",
        "limit",
        "join",
        "on",
        "as",
        "case",
        "when",
        "then",
        "else",
        "end",
        "sum",
        "count",
        "avg",
        "min",
        "max",
        "round",
        "desc",
        "asc",
        "left",
        "right",
        "inner",
        "outer",
        "distinct",
        "null",
        "is",
        "not",
        "in",
        "like",
        "with",
        "union",
        "all",
        "between",
        "coalesce"
    }

    aliases = {
        "s",
        "l",
        "p",
        "r",
        "e6"
    }

    unknown = [
        tok
        for tok in referenced_identifiers
        if tok not in sql_keywords
        and tok not in real_columns
        and tok not in real_tables
        and tok not in aliases
        and not tok.isdigit()
    ]

    # The notebook establishes the schema-conformance stage,
    # but the final gate relies on SQLite parsing/planning and
    # LLM relevance rather than treating every SQL token as
    # an independent schema error.

    # --------------------------------------------------------
    # Check 3: Parse and plan
    # --------------------------------------------------------

    try:

        cur.execute(
            f"EXPLAIN {candidate_sql}"
        )

        cur.fetchall()

    except sqlite3.Error as e:

        result["failed_check"] = (
            "parse_plan_dry_run"
        )

        result["details"] = (
            f"SQL failed to parse or plan: {str(e)}"
        )

        return result

    # --------------------------------------------------------
    # Check 4: LLM relevance
    # --------------------------------------------------------

    is_verified_track = (
        query_id is not None
        and query_id in query_library
    )

    if is_verified_track:

        track_context = """
This SQL is a pre-approved VERIFIED TEMPLATE.

It is intentionally broad and may return all sectors, categories,
or stages rather than filtering to just what the user asked.
A separate response-generation step will highlight the relevant
rows afterward. Do not fail it simply because it does not filter
to one particular sector, category, or stage.

Judge whether the underlying metric, tables, aggregation,
and business logic match the user's intent.
"""

    else:

        track_context = """
This SQL was freshly generated for this question.

It should be appropriately scoped and filtered to answer
the user's question directly.
"""

    relevance_prompt = f"""
### ROLE

You are a senior data validator.

Determine whether the SQL correctly answers the business
user's commercial lending credit-risk portfolio question.

### CONTEXT

{track_context}

### USER QUESTION

{user_question}

### CANDIDATE SQL

{candidate_sql}

### CHECK

Assess:

1. Correct tables and columns.
2. Correct aggregation and grouping, without double-counting loans
   across provisioning reporting_dates.
3. Correct business definitions (NPA, overdue, IFRS 9 stage).
4. Correct entity handling (e.g., 'Real Estate' = SEC_RE).
5. Correct answer shape (row-level vs aggregate, single date vs trend).
6. If this is a verified template, do not penalize it for returning
   a broader result set than the question's scope.

### OUTPUT

Return ONLY JSON:

{{
    "verdict": "yes" or "no",
    "confidence": 0.0 to 1.0,
    "reason": "one short sentence"
}}
"""

    relevance_response = evaluator_llm.invoke(
        relevance_prompt
    ).content.strip()

    json_match = re.search(
        r"\{.*\}",
        relevance_response,
        re.DOTALL
    )

    relevance_json = None

    if json_match:

        try:
            relevance_json = json.loads(
                json_match.group()
            )

        except json.JSONDecodeError:
            relevance_json = None

    if relevance_json is None:

        result["failed_check"] = (
            "llm_relevance"
        )

        result["details"] = (
            "Relevance check failed: "
            "validator response could not be parsed"
        )

        return result

    result["relevance_confidence"] = (
        relevance_json.get(
            "confidence",
            0.0
        )
    )

    if (
        relevance_json.get("verdict") == "no"
        or relevance_json.get(
            "confidence",
            0.0
        ) < 0.6
    ):

        result["failed_check"] = (
            "llm_relevance"
        )

        result["details"] = (
            "Relevance check failed: "
            + relevance_json.get(
                "reason",
                "unknown"
            )
        )

        return result

    # --------------------------------------------------------
    # Check 5: Verified template integrity
    # --------------------------------------------------------

    if is_verified_track:

        expected_sql = query_library[query_id]["sql"]

        try:

            expected_cols = [
                d[0]
                for d in cur.execute(
                    f"SELECT * FROM ({expected_sql}) LIMIT 0"
                ).description
            ]

            actual_cols = [
                d[0]
                for d in cur.execute(
                    f"SELECT * FROM ({candidate_sql}) LIMIT 0"
                ).description
            ]

            if len(expected_cols) != len(actual_cols):

                result["failed_check"] = (
                    "template_integrity"
                )

                result["details"] = (
                    f"Expected {len(expected_cols)} columns, "
                    f"got {len(actual_cols)}"
                )

                return result

        except sqlite3.Error as e:

            result["failed_check"] = (
                "template_integrity"
            )

            result["details"] = (
                f"Template integrity check failed: {str(e)}"
            )

            return result

    result["passed"] = True
    result["details"] = (
        "All validation checks passed"
    )

    return result


# ============================================================
# Tool 4: Retry Generation
# ============================================================

def retry_generation(
    user_question,
    failed_sql,
    error_message,
    schema_context
):

    retry_prompt = f"""
### ROLE

You are a senior SQL developer fixing a query that failed
validation.

### USER QUESTION

{user_question}

### FAILED SQL

{failed_sql}

### VALIDATION ERROR

{error_message}

### DATABASE SCHEMA

{schema_context}

### INSTRUCTIONS

1. Fix only the identified problem.
2. Preserve the original user intent.
3. Return read-only SELECT or WITH ... SELECT.
4. Use only valid tables and columns.
5. Use SQLite-compatible syntax.

### OUTPUT

Return ONLY the corrected SQL.

No markdown.
No comments.
No explanation.
"""

    revised_sql = llm.invoke(
        retry_prompt
    ).content.strip()

    revised_sql = re.sub(
        r"^```sql\s*|\s*```$",
        "",
        revised_sql,
        flags=re.IGNORECASE | re.MULTILINE
    ).strip()

    revised_sql = re.sub(
        r"^```\s*|\s*```$",
        "",
        revised_sql,
        flags=re.MULTILINE
    ).strip()

    return revised_sql


# ============================================================
# Tool 5: Query Execution
# ============================================================

def execute_query(
    validated_sql,
    db_connection
):

    result = {
        "dataframe": None,
        "reasonable": True,
        "warnings": []
    }

    df = pd.read_sql_query(
        validated_sql,
        db_connection
    )

    result["dataframe"] = df

    if df.empty:

        result["warnings"].append(
            "Query returned an empty result"
        )

    for col in df.select_dtypes(
        include="number"
    ).columns:

        if (
            (df[col] < 0).any()
            and "deviation" not in col.lower()
            and "change" not in col.lower()
        ):

            result["warnings"].append(
                f"Column {col} contains negative values"
            )

        if df[col].isnull().any():

            null_count = df[col].isnull().sum()

            if null_count > len(df) * 0.5:

                result["warnings"].append(
                    f"Column {col} has "
                    f"{null_count} null values"
                )

    if len(result["warnings"]) > 2:

        result["reasonable"] = False

    return result


# ============================================================
# Tool 6: Response Generation
# ============================================================

def generate_response(
    user_question,
    dataframe,
    route,
    query_id=None
):

    response_prompt = f"""
### ROLE

You are a credit-risk analyst writing a concise business
response for a commercial lending portfolio question.

### USER QUESTION

{user_question}

### QUERY RESULT DATA

{dataframe.to_string()}

### INSTRUCTIONS

1. Answer the specific question directly.
2. Do not dump the entire table.
3. If the user asked about a specific sector, borrower, IFRS 9
   stage, or DPD bucket, highlight only those rows.
4. Provide comparisons or rankings when useful.
5. State exact numbers from the data.
6. Flag notable credit risks, such as a high NPA ratio, a
   concentration in a sensitive sector, or a rising Stage 3
   or ECL trend.
7. Use professional language suitable for a credit risk
   committee memo.
8. Keep the response concise.
9. Interpret units from column names:
   - _usd = dollars
   - _million = millions of dollars
   - _pct = percentage
   - _count = count
   - _days = days

### OUTPUT

Return ONLY the natural-language response.

No headers.
No bullet points unless genuinely necessary.
"""

    response = llm.invoke(
        response_prompt
    ).content.strip()

    return response


# ============================================================
# Complete Workflow
# ============================================================

def run_workflow(
    user_question,
    status=None
):

    def report(message):

        if status is not None:
            status.write(message)

    log = {
        "user_question": user_question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": False,
        "executed_sql": None,
        "row_count": None,
        "confidence": None,
        "response": None
    }

    # --------------------------------------------------------
    # Step 1: Intent Classification
    # --------------------------------------------------------

    classification = classify_intent(
        user_question,
        VERIFIED_QUERY_LIBRARY
    )

    log["route"] = classification.get(
        "route",
        "generated"
    )
    log["query_id"] = classification.get(
        "query_id"
    )
    log["match_reason"] = classification.get(
        "match_reason"
    )

    if (
        log["route"] == "verified"
        and log["query_id"] not in VERIFIED_QUERY_LIBRARY
    ):

        log["route"] = "generated"
        log["query_id"] = None

    report(
        f"**Intent Classification:** "
        f"route=`{log['route']}`, "
        f"query_id=`{log['query_id']}`  \n"
        f"{log['match_reason']}"
    )

    # --------------------------------------------------------
    # Step 2: Query Construction
    # --------------------------------------------------------

    if log["route"] == "verified":

        candidate_sql = VERIFIED_QUERY_LIBRARY[
            log["query_id"]
        ]["sql"]

        report(
            "**Query Construction:** "
            "loaded from verified library"
        )

    else:

        candidate_sql = generate_query(
            user_question,
            DATABASE_SCHEMA
        )

        report(
            "**Query Construction:** "
            "generated fresh SQL"
        )

    log["candidate_sql"] = candidate_sql

    # --------------------------------------------------------
    # Step 3: Validation Gate
    # --------------------------------------------------------

    gate = validate_query(
        user_question,
        candidate_sql,
        conn,
        VERIFIED_QUERY_LIBRARY,
        log["query_id"]
    )

    log["gate_result"] = gate
    log["confidence"] = gate.get(
        "relevance_confidence"
    )

    report(
        f"**Validation Gate:** "
        f"passed=`{gate['passed']}`, "
        f"relevance_confidence="
        f"`{gate.get('relevance_confidence')}`"
    )

    # --------------------------------------------------------
    # Step 4: Retry Generated SQL
    # --------------------------------------------------------

    if (
        not gate["passed"]
        and log["route"] == "generated"
    ):

        report(
            f"Retrying: {gate['details']}"
        )

        candidate_sql = retry_generation(
            user_question,
            candidate_sql,
            gate["details"],
            DATABASE_SCHEMA
        )

        log["candidate_sql"] = candidate_sql
        log["retry_used"] = True

        gate = validate_query(
            user_question,
            candidate_sql,
            conn,
            VERIFIED_QUERY_LIBRARY,
            None
        )

        log["gate_result"] = gate
        log["confidence"] = gate.get(
            "relevance_confidence"
        )

        report(
            f"**Retry Validation Gate:** "
            f"passed=`{gate['passed']}`, "
            f"relevance_confidence="
            f"`{gate.get('relevance_confidence')}`"
        )

    # --------------------------------------------------------
    # Step 5: Escalation
    # --------------------------------------------------------

    if not gate["passed"]:

        log["escalated"] = True
        log["route"] = "escalate"
        log["confidence"] = "ESCALATED"

        log["response"] = (
            "Query could not be reliably resolved. "
            "Escalated to human analyst. "
            f"Failure: {gate['details']}"
        )

        report(
            f"**Escalated to human:** "
            f"{gate['details']}"
        )

        return {
            "log": log,
            "dataframe": None,
            **log
        }

    # --------------------------------------------------------
    # Step 6: Execute
    # --------------------------------------------------------

    log["executed_sql"] = candidate_sql

    exec_result = execute_query(
        candidate_sql,
        conn
    )

    df = exec_result["dataframe"]

    log["row_count"] = len(df)

    report(
        f"**Execute:** "
        f"{len(df)} rows returned"
    )

    if exec_result["warnings"]:

        report(
            f"Warnings: "
            f"{exec_result['warnings']}"
        )

    # --------------------------------------------------------
    # Step 7: Response Generation
    # --------------------------------------------------------

    response = generate_response(
        user_question,
        df,
        log["route"],
        log["query_id"]
    )

    log["response"] = response

    log["confidence"] = gate.get(
        "relevance_confidence"
    )

    report(
        f"**Response Generation:** "
        f"confidence=`{log['confidence']}`"
    )

    return {
        "log": log,
        "dataframe": df,
        **log
    }


# ============================================================
# Streamlit User Interface
# ============================================================

st.title(
    "🏦 Credit Risk Query Engine"
)

st.caption(
    "Ask natural-language questions about the commercial lending "
    "portfolio: sector exposure, NPAs, delinquency, IFRS 9 staging, "
    "ratings, and provisioning."
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This application routes credit-risk questions through "
        "a verified SQL template library when possible, or "
        "generates fresh SQL for novel questions. Every query "
        "passes through validation before database execution, "
        "and unresolved queries are escalated to a human analyst."
    )

    st.subheader("Example questions")

    st.markdown(
        "- How much of our book is in real estate, and how much "
        "of it is non-performing?\n"
        "- Show me how our overdue book is distributed across "
        "DPD buckets.\n"
        "- List every restructured loan, flag which are impaired, "
        "and show the overall impaired percentage.\n"
        "- Show me the average interest rate by sector, sorted "
        "from highest to lowest.\n"
        "- How has the Stage 3 book size and its expected credit "
        "loss moved over the last four quarters?"
    )


# ============================================================
# User Input
# ============================================================

user_question = st.text_input(
    "Your question",
    placeholder=(
        "e.g. Show me how our overdue book is distributed "
        "across DPD buckets."
    )
)

show_trace = st.checkbox(
    "Show pipeline trace",
    value=True
)

submitted = st.button(
    "Run query",
    type="primary"
)


# ============================================================
# Run Application
# ============================================================

if submitted and user_question.strip():

    trace_container = (
        st.status(
            "Running pipeline...",
            expanded=show_trace
        )
        if show_trace
        else None
    )

    try:

        output = run_workflow(
            user_question=user_question,
            status=trace_container
        )

    except Exception as e:

        if trace_container is not None:

            trace_container.update(
                label="Pipeline error",
                state="error"
            )

        st.error(
            f"Pipeline failed: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # Pipeline Status
    # --------------------------------------------------------

    if trace_container is not None:

        trace_container.update(
            label=(
                "Pipeline complete"
                if not output["escalated"]
                else "Escalated to human review"
            ),
            state=(
                "complete"
                if not output["escalated"]
                else "error"
            )
        )

    st.divider()

    # --------------------------------------------------------
    # Escalation
    # --------------------------------------------------------

    if output["escalated"]:

        st.warning(
            output["response"]
        )

        with st.expander(
            "Validation details"
        ):

            st.json(
                output["gate_result"]
            )

    # --------------------------------------------------------
    # Successful Result
    # --------------------------------------------------------

    else:

        confidence = output["confidence"]

        if isinstance(
            confidence,
            (int, float)
        ):

            if confidence >= 0.8:

                badge = "🟢"

            elif confidence >= 0.6:

                badge = "🟡"

            else:

                badge = "🔴"

            confidence_display = (
                f"{confidence:.2f}"
            )

        else:

            badge = "⚪"
            confidence_display = str(
                confidence
            )

        st.subheader("Answer")

        st.write(
            output["response"]
        )

        st.caption(
            f"{badge} Confidence: "
            f"{confidence_display}  ·  "
            f"Route: {output['route']}  ·  "
            f"Query ID: {output['query_id']}  ·  "
            f"Rows returned: {output['row_count']}"
        )

        # ----------------------------------------------------
        # Underlying Data
        # ----------------------------------------------------

        if output["dataframe"] is not None:

            with st.expander(
                "View underlying data"
            ):

                st.dataframe(
                    output["dataframe"],
                    use_container_width=True
                )

        # ----------------------------------------------------
        # Executed SQL
        # ----------------------------------------------------

        with st.expander(
            "View executed SQL"
        ):

            st.code(
                output["executed_sql"],
                language="sql"
            )

else:

    if submitted:

        st.info(
            "Please enter a question first."
        )
