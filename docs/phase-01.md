## After reading build-log.md start reading this:

### Day 1 (read until Day 10)
---

### 1) Hybrid vLLM Setup & Execution Log

* **Colab GPU Offloading (vLLM):**
* Offloaded `Qwen/Qwen2.5-3B-Instruct-AWQ` to a free Google Colab T4 GPU to bypass local GPU requirements.
* **Technique:** Served vLLM on port `8000` in Colab and exposed it via an HTTPS tunnel using `cloudflared` (Cloudflare Tunnel).
* **Notebook Reference:** Full notebook setup and execution cell saved in `playbook/playbook.ipynb`.
* **Compose adjustment:** Updated local `.env` (`VLLM_BASE_URL=https://<subdomain>[.trycloudflare.com/v1](https://.trycloudflare.com/v1)`) and commented out the local `vllm` service dependency in `docker-compose.yml`.



### 2) Sequential Execution & Test Commands

```bash
# 1. Spin up core database and admin UI locally
docker compose up -d postgres adminer

# 2. Run one-off database migration & seed 1,000 property records
docker compose --profile tools run --rm migrate

# 3. Verify database seeding (expected output: 1000)
docker exec -it realestate-postgres psql -U realestate -d realestate -c "SELECT count(*) FROM units;"
```

**Check Adminer UI for database on:**

http://localhost:8082/

### 3) For rest of commands


**- run rest of services**
```bash
# 1. Spin up local gateway & observability stack (without local vLLM)
docker compose up -d gateway redis qdrant prometheus grafana otel-collector
[or]
docker compuse up -d # and will run all except what is actually running now
```

**- Dynamic vLLM Endpoint Update (When Colab Restarts)**
```bash
Update VLLM_BASE_URL in .env & compose.yml **with the new Cloudflare link**
VLLM_BASE_URL=https://<new-subdomain>[.trycloudflare.com/v1](https://.trycloudflare.com/v1)
docker compose up -d gateway # Reload Gateway service
```

**- Verification Checklist**

* 1) Terminal Commands
Check service health: docker compose ps (All should show Up / Healthy)
Verify 1,000 seeded DB rows using:

```bash
docker exec -it realestate-postgres psql -U realestate -d realestate -c "SELECT count(*) FROM units;"
```

* 2) Test Gateway health:
```bash
curl http://localhost:8081/health
```

* 3) Test Gateway -> Colab vLLM pass-through:

```bash
curl http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-3B-Instruct-AWQ","messages":[{"role":"user","content":"Hi"}]}'
```

* 4) Verify OTel Collector logs:
```bash
docker compose logs otel-collector
```

* 5) Local Browser Endpoints
    * Gateway Swagger:

        http://localhost:8081/docs

    * Adminer (Postgres UI):
    
        http://localhost:8082 (System: Postgres | Server: postgres | User/Pass/DB: realestate)
    
    * Qdrant Dashboard: 
    
        http://localhost:6333/dashboard
    
    * Prometheus Targets: 
    
        http://localhost:9090/targets (Verify target state is UP)
    
    * Grafana: 
    
        http://localhost:3000 (Login: admin / admin)

**- Cleanup Compose containers**
```bash
docker compose down
```