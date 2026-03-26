# Customer Support Triage (OpenEnv)

This project implements an OpenEnv environment where an AI agent resolves customer support tickets by interacting with a simulated workflow: querying a mocked customer database, issuing refunds with strict parameter matching, sending resolution emails, or escalating tickets when needed.

## Motivation (Real-World Utility)

Customer support operations are a high-impact, real-world use case for evaluating LLM agents. This environment is valuable because it tests:

- Multi-step reasoning across sequential decisions
- API/tool usage (`query_db`, `issue_refund`, `send_email`, `escalate`)
- Strict parameter correctness (for example, refund amount and currency matching)
- Task completion quality via deterministic grading

## Observation & Action Spaces

### Observation Schema

The environment returns observations using the `Observation` model:

```json
{
	"ticket_id": "string",
	"customer_message": "string",
	"database_state": {
		"key": "string"
	},
	"previous_action_result": "string"
}
```

### Action Schema

The agent must return actions using the `Action` model:

```json
{
	"action_type": "query_db | issue_refund | send_email | escalate",
	"parameters": {
		"key": "string"
	}
}
```

### Valid Actions

1. `query_db`
	 - Purpose: Reveal hidden customer/account/order details in the mocked database state.
	 - Typical parameters: `{}`

2. `issue_refund`
	 - Purpose: Attempt a refund using exact required values.
	 - Typical parameters: `{"amount": "59.99", "currency": "USD"}`

3. `send_email`
	 - Purpose: Send a final customer-facing message and resolve the ticket.
	 - Typical parameters: `{"body": "Your refund has been processed..."}`

4. `escalate`
	 - Purpose: Escalate to a human/department for issues requiring specialized handling.
	 - Typical parameters: `{"reason": "Password reset required", "target": "IT_Support"}`

## Tasks

The environment includes three deterministic evaluation tasks:

- **Easy (`task_easy`)**
	- Password recovery scenario.
	- Expected behavior: escalate to `IT_Support`.

- **Medium (`task_medium`)**
	- Warranty status inquiry.
	- Expected behavior: `query_db`, then `send_email` containing the exact warranty expiration date from database state.

- **Hard (`task_hard`)**
	- Broken monitor refund flow.
	- Expected behavior: `query_db`, `issue_refund` with exact amount/currency, then `send_email` confirmation.

## Setup & Usage (Docker)

Build and run locally with Docker:

```bash
docker build -t openenv-triage .
docker run -p 7860:7860 openenv-triage
```

Once running, the API is available at:

- `http://localhost:7860`
- Swagger docs: `http://localhost:7860/docs`

## Running the Baseline

The baseline script (`baseline.py`) uses `gpt-4o-mini` through the OpenAI Python SDK and interacts with the local FastAPI server.

1. Set your API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

2. Run the baseline agent:

```bash
python baseline.py
```

The script will:

- Fetch all tasks from `/tasks`
- Run one episode per task
- Query `/grader` for final score
- Print per-task and aggregate baseline results

## Project Files

- `openenv.yaml`: Environment metadata
- `models.py`: Pydantic schemas (`Observation`, `Action`, `Reward`)
- `env.py`: Core environment state machine (`reset`, `step`, `state`)
- `tasks.py`: Task definitions and deterministic grader
- `main.py`: FastAPI server and OpenEnv routes
- `baseline.py`: LLM baseline evaluation loop
- `requirements.txt`: Python dependencies
- `Dockerfile`: Containerized deployment for Hugging Face Spaces/local use