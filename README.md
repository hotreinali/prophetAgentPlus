# Extended ProphetAgent: Automated Flow-Path Generation via Prompt Engineering
This repository contains an extended version of ProphetAgent (https://github.com/ProphetAgent/Home), redesigned to enable fully automated flow-path generation for React Native applications using LLM-based prompt engineering and Neo4j-driven navigation analysis.

The extension removes ProphetAgent’s original dependency on manually authored flow_path.json files by introducing an LLM-controlled Flow Path Automation Module capable of generating navigation paths through Cypher queries.

This project evaluates how different prompt engineering strategies (zero-shot, structured/few-shot, context-rich/CoT) influence the quality of test-case generation across multiple LLMs: GPT-5-mini, GPT-4o, GPT-4o-mini.

---

## Key Contributions
1. Automated Flow-Path Generation
- Eliminates manual flow_path.json creation.
- Uses LLM-generated Cypher queries to retrieve navigation paths from Neo4j.

2. Prompt Engineering Integration
| Prompt Levvel | Description | Purpose |
|-------|-----------|-----------|
| **L1 – Basic (Zero-shot)** | Minimal instructions | Baseline performance |
| **L2 – Structured (Few-shot)** | Schema rules, directional edges | Improves consistency & validity |
| **L3 – Context-rich (Chain-of-thought guided)** | Domain context & alternation constraints | Maximizes semantic accuracy |

3. Multi-Model Evaluation
Evaluated from:
- GPT-5-mini
- GPT-4o
- GPT-4o-mini

---

## Future Work
- Extending execution to iOS simulators
- Adding execution feedback loops for self-refining prompts
- Automatic Detox/Appium test script synthesis

