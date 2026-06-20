# Local Agent Eval

A deterministic AI-style workflow that runs without model credentials.

It teaches the shape of an agent loop:

- classify input
- call a local tool
- produce structured output
- evaluate behavior against examples

## Run

```bash
python3 projects/local-agent-eval/app.py
```

## Test

```bash
python3 projects/local-agent-eval/test_app.py
```

## Next Exercises

1. Add a new tool for priority scoring.
2. Add expected outputs in a JSON file.
3. Replace the deterministic planner with a model call behind an interface.
4. Keep the same eval before and after adding an external model.

