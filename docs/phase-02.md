# Phase-2:
**including:**
* VectorDB ingestion & RAG
* Build Agent
* Integerate Services (OTel, Promethesus, Grafana, Redis)

## a) VectorDB ingestion & RAG


## b) Build Agent
Keep the agent package flat under  services\agent\ ; that keeps paths shallow and responsibilities clear. Your repo already has  agent_runner.py ,  ingest_units.py , and  ingest_docs.py . Keep the ingestion scripts focused on indexing; treat  agent_runner.py  as the current prototype, not a file to overwrite blindly.

services/
  agent/
    agent_runner.py       # current prototype / eventual entry point
    config.py             # model URL/name and agent limits
    llm_client.py         # only communicates with vLLM
    schemas.py            # typed conversation, tool-call, and result shapes
    prompts.py            # system instructions and tool descriptions
    memory.py             # short-term conversation-window interface
    tools.py              # tool registry, argument validation, dispatch
    orchestrator.py       # model ↔ tool loop and stopping conditions
    ingest_units.py       # existing: inventory indexing
    ingest_docs.py        # existing: document indexing
tests/
  test_agent_orchestrator.py
  test_agent_tools.py

Build in this order:

1.  llm_client.py : make one ordinary chat-completion call to your existing vLLM endpoint.
2.  schemas.py  and  tools.py : define two read-only tools— search_units  and  search_documents —with validated arguments. The model may request a tool; only your code decides whether and how to execute it.
3.  memory.py : start with a bounded conversation window for one session. Keep the interface replaceable; later we can back it with Redis or Postgres. Don’t add long-term or semantic memory yet.
4.  orchestrator.py : implement the control loop: call model → validate tool request → execute tool → add result to conversation → call model again. Cap tool rounds (for example, 3), and handle invalid calls and tool errors explicitly.
5. Add tests with a fake model client and fake tools, so the control-flow tests do not depend on vLLM being online.

Keep these as agent logic only for now—no FastAPI routes, MCP servers, observability, Redis, or deployment changes. Once you’ve written the agent files, bring them back for review; then we’ll connect the read-only tools to your existing Qdrant collections and Postgres. I’ve set the tracked workstream to basic RAG + Agent loop; evaluation remains paused until the build is ready.
