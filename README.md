#   Geopolitical Capacity Limit Squeezer & AI Agent Loop

An enterprise-grade, full-stack decision-intelligence architecture that models, visualizes, and resolves supply chain capacity crises. This repository integrates strict mathematical rigor (linear algebra, limit calculus, and stochastic dynamic programming) with agentic orchestration (LangGraph, pgvector/PostgreSQL, and Gemini 2.5) and an asynchronous Spring Boot REST API gateway.

## 🗺️ System Architecture

```

          [ React Frontend / API Client ]
                         │
                         ▼ (HTTP GET /api/logistics/run-simulation)
          [ Spring Boot REST Controller ]
                         │
                         ▼ (Asynchronous Java ProcessBuilder Task)
          [ LangGraph State Machine Engine ]
            ├── analyst_node (Linear Algebra & Route Vectors)
            ├── memory_retrieval_node (pgvector Semantic Search)
            ├── coo_node (Gemini 2.5 Strategy synthesis)
            └── archivist_node (Generative Memory Committer)
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
   [ pgvector Docker Container ]  [ Local File System ]
   (Relational Postgres DB)        (Logarithmic PNG Chart & JSON)
```

## 🎯 Project Objectives

Deterministic Resource Pooling: Treat transportation corridors as coordinate coordinates in multi-dimensional space, combining throughput, risk, and base costs dynamically.

Preemptive Infrastructure Safeguards: Apply the Calculus Squeeze Theorem to calculate physical throughput bottlenecks and prevent exponential operational cost explosions before they manifest.

Memoryless Environmental Physics: Model unstable, non-stationary logistics networks as Markov chains to isolate today's geopolitical realities from historical statistical biases.

Recursive Strategy Optimization: Resolve optimal operational policies over infinite horizons using discounted recursive state updates.

Generative Memory Pipelines: Architect self-correcting agents that check their own historical decisions using high-speed semantic database indexing.

##  📐 Mathematical Foundations

1. Multi-Dimensional Vector Spaces (Linear Algebra)

Logistics routes are modeled as coordinate vectors $\mathbf{u}$ in a $3$-dimensional vector space $\mathbb{R}^3$, where each coordinate represents critical operating parameters:

$$
\mathbf{u} = \begin{bmatrix} T \\\\ C \\\\ R \end{bmatrix} = \begin{bmatrix} \text{Throughput (tons/day)} \\\\ \text{Base Operating Cost (k\$)} \\\\ \text{Geopolitical Risk Index (1-10)} \end{bmatrix}
$$

Resource Pooling (Vector Addition): When corridors are merged, their throughput, base costs, and risks pool continuously:

$$\mathbf{w} = \mathbf{u} + \mathbf{v} = \begin{bmatrix} T_u + T_v \\ C_u + C_v \\ R_u + R_v \end{bmatrix}$$

Intensity Scaling (Scalar Multiplication): Scaling a corridor's operational footprint (e.g., adding truck fleets) scales all vector metrics simultaneously by a capacity factor $c$:

$$c\mathbf{u} = \begin{bmatrix} c T_u \\ c C_u \\ c R_u \end{bmatrix}$$

2. Squeeze Theorem Boundary Analysis (Calculus 1)

Let $c$ represent our capacity scaling factor. As cargo volume approaches the physical structural ceiling of the highway network ($M = 5.0$), congestion causes empirical operating costs $C(c)$ to blow up asymptotically:

$$\lim_{c \to M^-} C(c) = \lim_{c \to 5.0^-} \frac{k}{M - c} = \infty$$

To guarantee operational stability under random noise, our analyst node establishes lower linear and upper parabolic boundaries to "squeeze" and isolate the safe operating envelope:

$$L(c) \le C(c) \le U(c)$$

Where:

Lower Linear Bound: $L(c) = a \cdot c + b$

Upper Parabolic Bound: $U(c) = p \cdot c^2 + q$

3. State Transitions & Markov Chains (Probability)

The supply chain is modeled as a memoryless Markov Decision Process (MDP). The probability of transitioning to state $s'$ tomorrow depends only on the observed state $s$ today and the chosen action $a$ tonight:

$$P(S_{t+1} = s' \mid S_t = s, A_t = a, \dots) = P(S_{t+1} = s' \mid S_t = s, A_t = a)$$

4. Recursive Value Resolution (Bellman Equation)

To evaluate the long-term risk-adjusted cost of active policies over an infinite horizon, we collapse the infinite future sequence using the recursive Bellman Equation:

$$Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q(s', a')$$

🛠️ Step-by-Step Execution Guide

1. Spin Up the Database

docker compose up -d --build

2. Configure Python & LangGraph

# Add dependencies
uv add langgraph langchain-google-genai langchain-postgres psycopg-binary psycopg

# Configure environment
export GEMINI_API_KEY="your-api-key"
unset GOOGLE_CLOUD_PROJECT

# Execute
```bash
uv run langgraph_orchestrator.py
```

3. Launch Spring Boot

```bash
mvn spring-boot:run
curl -i http://localhost:8080/api/logistics/run-simulation
```