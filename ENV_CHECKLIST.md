# Environment Initialization Checklist

Use this checklist before running `ingest_fda.py` or the Streamlit app.

## 1. Python environment

- [ ] Python 3.10+ installed (`python3 --version`)
- [ ] Virtual environment created:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate   # macOS / Linux
  ```
- [ ] Dependencies installed:
  ```bash
  pip install -r requirements.txt
  ```

## 2. Environment variables

- [ ] Copy the template: `cp .env.template .env`
- [ ] **Supabase** — create a project at [supabase.com](https://supabase.com)
  - [ ] `SUPABASE_URL` from Project Settings → API
  - [ ] `SUPABASE_ANON_KEY` for client-side / Streamlit reads
  - [ ] `SUPABASE_SERVICE_ROLE_KEY` for server-side ingestion (keep secret)
- [ ] **OpenAI** — create an API key at [platform.openai.com](https://platform.openai.com)
  - [ ] `OPENAI_API_KEY` set in `.env`
- [ ] **openFDA** (optional) — request a key at [open.fda.gov/apis/authentication](https://open.fda.gov/apis/authentication/)
  - [ ] `OPENFDA_API_KEY` set if you need higher rate limits

## 3. Local directories

The ingest script creates these automatically, but you can pre-create them:

- [ ] `data/raw/` — downloaded FDA label JSON
- [ ] `logs/` — pipeline run logs

```bash
mkdir -p data/raw logs
```

## 4. Verify openFDA connectivity (no Supabase required)

- [ ] Run the sample ingest:
  ```bash
  python ingest_fda.py
  ```
- [ ] Confirm output files exist:
  - [ ] `data/raw/fda_labels_sample.json`
  - [ ] `logs/ingest_fda.log`
- [ ] Review the console summary for field names and record count (expect 50)

## 5. Supabase schema (next step)

After validating the API payload shape, create a table to store labels. Example columns:

| Column        | Type   | Notes                          |
|---------------|--------|--------------------------------|
| id            | uuid   | primary key, default gen_random_uuid() |
| set_id        | text   | unique SPL set identifier      |
| brand_name    | text   | from `openfda.brand_name`      |
| generic_name  | text   | from `openfda.generic_name`    |
| product_type  | text   | e.g. HUMAN PRESCRIPTION DRUG   |
| effective_time| text   | label effective date           |
| raw_payload   | jsonb  | full openFDA record            |
| ingested_at   | timestamptz | default now()             |

- [ ] Table created in Supabase SQL editor
- [ ] Row Level Security policies defined for your app roles
- [ ] Service role key tested from a backend script (not exposed in Streamlit)

## 6. Security

- [ ] `.env` is listed in `.gitignore` and never committed
- [ ] `SUPABASE_SERVICE_ROLE_KEY` used only in server-side scripts
- [ ] Streamlit app uses `SUPABASE_ANON_KEY` with RLS, not the service role

## 7. Streamlit (when ready)

- [ ] App entry point: `streamlit run app/main.py`
- [ ] `.env` loaded via `python-dotenv` or Streamlit secrets
