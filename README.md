# E-Commerce AI Support Agent — Refund Approval & Evaluation Extension

A portfolio extension of [Crate — AI Support Agent](https://github.com/m-peker/ecommerce-ai-agent), an open-source e-commerce customer-support agent. This project extends the existing LangGraph, FastAPI, Streamlit and SQLite application with order cancellation, refund requests, human approval for high-value refunds, deterministic tool routing and a repeatable evaluation suite.

> **Scope:** Local demonstration with seeded orders and a configured demo customer identity. Refund approval updates application records; **no real payment or refund transfer** occurs. This is not a production-ready authentication or payments system.

## What I added

The upstream project already included a LangGraph triage/tool/response pipeline, order and shipment lookups, returns tools, FastAPI, Streamlit and SQLite. My extension focuses on the following additions:

- **Order cancellation:** Added a `cancel_order` service and agent tool, with order ownership and status checks. Only eligible, confirmed orders can be cancelled.
- **Refund requests:** Added a SQLite `refunds` table and `request_refund` workflow with checks for demo customer ownership, eligibility, time window and duplicate requests. In this demo, refunds up to $100 are marked `approved`; higher-value refunds require human review.
- **Human-in-the-loop approval:** Built a separate LangGraph approval workflow using `interrupt()`, SQLite-backed checkpoints and `Command(resume=...)`. Added API endpoints for starting approval and submitting an `approve` / `reject` decision. The chat flow can initiate a pending approval workflow automatically.
- **Deterministic routing for sensitive operations:** After LLM intent classification, explicit cancellation and refund requests with a valid order ID are routed directly to their corresponding business tools. Other intents retain the original LLM-based tool selection flow. Missing order IDs do not trigger state-changing tools.
- **Evaluation and regression tests:** Created a 40-case evaluation dataset and an evaluation runner with isolated SQLite test data, tool-selection checks, database state assertions and human-approval checks. Added pytest coverage for cancellation, refunds, approval graphs, API approval and routing.

## Architecture

```text
Customer (Streamlit or HTTP)
           |
           v
       FastAPI /chat
           |
           v
    LangGraph triage (LLM)
           |
           v
   Intent + extracted order ID
           |
           +-------------------------------+
           |                               |
   cancel_order / refund_request      Other supported intents
           |                               |
   Deterministic Python routing       LLM tool selection
           |                               |
           +---------------+---------------+
                           |
                     Business tools
                           |
                 SQLite service checks
                           |
          High-value refund pending approval?
                           |
                   Yes --> Approval graph
                           |     interrupt()
                           |     SQLite checkpoint
                           |     API human decision
                           |     Command(resume=...)
                           v
                    Response generation
```

Business rules are enforced in the service layer rather than by prompts alone.

## Implemented API endpoints

The routes are mounted under `/api/v1`:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/chat` | Customer message and agent response |
| `POST` | `/api/v1/refunds/{order_id}/approval/start` | Start or inspect a pending approval workflow |
| `POST` | `/api/v1/refunds/{order_id}/approval/decision` | Resume approval with `approve` or `reject` |

Example decision body:

```json
{"decision": "approve"}
```

**Important:** The approval endpoints use demo identity checks, not real administrator authentication. Do not expose this application publicly without adding authentication, authorization and appropriate concurrency safeguards.

## Evaluation

The same 40 scenarios were run in three development iterations: 10 normal requests, 10 policy cases, 8 human-approval cases, 8 ambiguous requests, and 4 error cases. Each iteration executed all 40 scenarios without a runner error; **32 scenarios had an automatic success verdict**, while **8 ambiguous scenarios were reserved for manual response review**.

| Metric | Baseline | Prompt rules | Deterministic routing |
| --- | ---: | ---: | ---: |
| Automatically scored task success | 30/32 (93.75%) | 29/32 (90.63%) | **32/32 (100%)** |
| Tool routing / selection accuracy | 30/32 (93.75%) | 29/32 (90.63%) | **32/32 (100%)** |
| Intent classification accuracy | 32/32 | 32/32 | 32/32 |
| Human-approval compliance | 8/8 | 8/8 | 8/8 |
| Detected policy violations | 0 | 0 | 0 |
| Execution errors | 0 | 0 | 0 |

**Interpretation:** The final run improved the automatically scored task success metric by **6.25 percentage points** relative to baseline. The final tool-routing result includes **Python-defined routing**, so it is not a measurement of independent LLM tool-choice accuracy. These figures refer to this fixed evaluation set and individual runs, not a general reliability guarantee. The policy-violation counter checks the rules covered by the runner; it is not a security audit.

### Failure analysis

- In the baseline, `TC011` and `TC014` did not call the expected cancellation tool for ineligible orders.
- Prompt-only changes corrected `TC011`, but `TC015` and `TC020` regressed; `TC014` still failed.
- Deterministic routing resolved all three of those cases in the final run (`TC014`, `TC015`, `TC020`) while keeping the recorded HITL checks at 8/8.

**Manual review notes:** Ambiguous requests did not change the database in the final run. The agent still overstates its ability to update delivery information in `TC032`, and `TC036` checks return eligibility before clarifying the customer's desired action. Those eight cases are not counted as automatic successes.

Raw results: `evaluation/results/baseline_20261118T140213Z.json`, `baseline_20261118T141503Z.json`, `baseline_20261118T142651Z.json` (when included in this repository).

## Setup (Windows PowerShell)

Prerequisites: Python 3.11+ and an OpenAI API key.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install langgraph-checkpoint-sqlite
Copy-Item .env.example .env
```

Configure your own API key in `.env` and, for the demo-only cancellation/refund tools, set the account used for the sample orders:

```dotenv
OPENAI_API_KEY=your-own-key
OPENAI_MODEL=gpt-4o
DEMO_CUSTOMER_EMAIL=marcus@example.com
```

Run the application:

```powershell
python run.py
```

- Chat UI: http://localhost:8501
- FastAPI documentation: http://localhost:8000/docs

### Tests and evaluation

```powershell
python -m pytest tests/ -q
python -m evaluation.run_evaluation --validate
python -m evaluation.run_evaluation --limit 3 --no-save
python -m evaluation.run_evaluation
```

The full evaluation invokes the OpenAI API and may incur charges. It uses isolated scenario data rather than modifying the regular demo database. For comparable results, keep `evaluation/scenarios.json` unchanged across runs.

If your dependency file does not already include `langgraph-checkpoint-sqlite`, add it before sharing the repository so other users can reproduce the approval workflow.

## Key files

```text
src/agent/nodes.py                # Intent classification and hybrid tool routing
src/agent/tools.py                # LangChain tool integrations
src/agent/refund_approval.py      # Human-approval graph and SQLite checkpointer
src/api/routes.py                 # Chat and approval endpoints
src/services/order_service.py     # Order cancellation checks
src/services/returns_service.py   # Refund workflow and rules
src/services/database.py          # SQLite tables, including refunds
evaluation/scenarios.json         # 40 evaluation cases
evaluation/run_evaluation.py      # Isolated execution and scoring
evaluation/results/              # Saved round-level JSON artifacts
tests/                           # Unit and integration tests
```

## Limitations and implementation notes

- The configured `DEMO_CUSTOMER_EMAIL` represents a simulated customer, not a verified user session; approval endpoints have no administrator login.
- `approved` is an application workflow status, not a payment-provider payout.
- The current refund eligibility implementation uses order creation dates in its 14-day check; the upstream policy text describes a delivery-based period. A production implementation should align these rules with recorded delivery dates.
- SQLite-backed approval checkpoints are useful for demonstrating pause and resume, but additional access control, transactional consistency, concurrent-request handling and audit logging would be needed for deployment.

## Attribution and license

This repository is an **extension of**, not an original implementation of, [m-peker/ecommerce-ai-agent](https://github.com/m-peker/ecommerce-ai-agent). The original application architecture, initial business tools, seed data and UI are credited to its author. My contributions are the cancellation/refund extensions, human-approval workflow, deterministic tool routing, additional tests and evaluation pipeline described above.

The upstream repository identifies its license as MIT. Preserve the upstream `LICENSE` and applicable notices when publishing a fork or derivative.
