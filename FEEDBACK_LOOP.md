# Feedback Loop Overview

The misinformation detector includes a backward propagation feedback loop that
uses a local LLM judge (via Ollama) and a fuzzy logic engine to improve model
behaviour over time.

Before training or running feedback cycles, ensure Ollama is running and the
required model is available:

```bash
ollama pull llama3
ollama serve
```

Subsequent phases will extend this document with a full step-by-step
description of the feedback cycle once all components are implemented.

