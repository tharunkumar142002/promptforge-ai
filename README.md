# 🧠 PromptForge AI — Prompt Engineering Studio

A portfolio-ready, interactive **Prompt Engineering** project built with Python, Streamlit, and a Groq-hosted LLM. Complete application code, evaluation data, tests, CI, and instructions for free-tier deployment are included.

> **Status:** Runs locally after dependencies are installed; live generation requires your own Groq API key and network access. Deploying to a public URL is a manual action you perform with GitHub + Streamlit Community Cloud. No deployment or paid account is included.

## Features

- 🧪 **Prompt Lab** — zero-shot, role-based, few-shot, JSON, context-grounded, constraints-first
- 🔎 **Preview** — see the system/user/example messages *before* requesting a completion
- ⚖️ **A/B Compare** — run two techniques on one task and compare responses, latency, and token usage
- ✨ **Optimizer** — use an LLM to rewrite weak prompts into stronger prompts
- 📊 **Evaluation** — upload a small labeled CSV and calculate transparent keyword coverage
- 🕘 **Session history** — inspect and export up to 50 generations (session-only)
- 📚 **Handbook** — in-app prompt-engineering explanations
- 🔐 **Key handling** — `.env` or Streamlit secrets, plus optional password-masked input in sidebar
- ✅ **Tests + CI** — pytest, offline mocked model tests, GitHub Actions

### Stack

| Layer | Tool |
| --- | --- |
| UI | Streamlit |
| API SDK | `groq` |
| Model | Llama 3.3 70B Versatile or Llama 3.1 8B Instant (available model/access may change) |
| Configuration | `python-dotenv` or Streamlit secrets |
| Testing | `pytest` |
| Deployment | GitHub + Streamlit Community Cloud |

## Get started (Windows PowerShell, Python 3.12)

1. Extract this ZIP under `C:\projects\PromptForge_AI_Project` (or any path).
2. Open that folder in VS Code and open a PowerShell terminal.
3. Run these commands:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

4. Create a free Groq API key at https://console.groq.com/keys. Copy `.env.example` to `.env` and **replace** the placeholder:

```powershell
Copy-Item .env.example .env
notepad .env
```

   Your `.env` should contain one line like `GROQ_API_KEY=gsk_your_actual_key` (do **not** use the literal example). Alternatively, skip `.env` and paste your key into the app's masked sidebar field. **Never commit `.env`, secret files, or API keys.**

5. Run unit tests and start the website:

```powershell
python -m pytest -q
python scripts/check_setup.py
python -m streamlit run app.py
```

Open http://localhost:8501. **Prompt preview works without a key; calls to the LLM require a key.**

### Project layout

```text
PromptForge_AI_Project/
├── app.py                        # Complete Streamlit UI
├── promptforge/
│   ├── __init__.py
│   ├── engine.py                 # Prompt techniques and validation
│   ├── llm.py                    # Groq model call + usage/error handling
│   └── evaluation.py             # Evaluation CSV + keyword coverage
├── data/sample_evaluation.csv
├── tests/
│   ├── test_engine.py
│   ├── test_evaluation.py
│   └── test_llm.py
├── scripts/check_setup.py
├── .streamlit/config.toml        # Theme
├── .streamlit/secrets.toml.example
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── DEPLOYMENT.md
├── LICENSE
└── README.md
```

## Example exercises

1. **Zero-shot:** "Describe a vector database in under 80 words."
2. **Role-based:** Select an experienced GenAI engineer; ask for a beginner-friendly RAG explanation.
3. **Few-shot:** Use `good => positive` and `bad => negative` as examples; ask to classify a new word.
4. **Structured JSON:** Ask to summarize an issue and receive `summary`, `key_points`, and `action_items`.
5. **Context-grounded:** Paste policy text, then ask a question about it. If the answer is absent, the model is instructed to say so (verify its behavior; it's not guaranteed).
6. **Evaluation:** Upload `data/sample_evaluation.csv`, select a technique, and compare keyword pass rates.

## What evaluation does *not* measure

The eval checks **whether specified strings appear in model outputs** (case-insensitive). A passing result does not prove correctness, usefulness, safety, or factual grounding. Expand the test suite with human judgments and task-specific scoring before using this for consequential work.

## Safety, quotas, and privacy

- Free Groq API access has quota/rate limits that can change. Not all models or limits may be available for every account.
- The UI caps requests at 30 attempted calls per browser session; evaluation caps runs at 5 cases. **These are convenience controls, not server-side rate limiting** and will not prevent deliberate misuse if you host a shared key publicly.
- For a public Streamlit demo, prefer asking visitors to use their **own** API keys rather than exposing your organization's quota through a shared key. Never publish a key in your GitHub repo.
- Prompts travel to Groq when you generate. Do not submit passwords, health records, personal identifiers, or confidential materials. Review provider terms and data settings.
- Local session history is not persistent and is not user-authenticated. This app is a portfolio demonstration, not a hardened multi-tenant service.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for exact GitHub push and Streamlit Community Cloud instructions.

## Source references

- Groq: https://console.groq.com/docs/text-chat
- Groq models: https://console.groq.com/docs/models
- Groq rate limits: https://console.groq.com/docs/rate-limits
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Streamlit secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

## License

MIT (see LICENSE). Third-party libraries and models retain their own licenses/terms.
