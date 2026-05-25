# scripts/dev — developer utilities

This directory holds sandbox / development helpers that are **not part of
the runtime package**. Two roles:

1. **Alternative test runners.** Each `verify*.py` is a stdlib-`unittest`
   script that exercises a slice of the codebase without needing
   `pytest`. Useful when:
   - You're working in a constrained environment (CI sandbox, fresh
     Raspberry Pi without dev deps).
   - You want a quick smoke run of one subsystem without invoking the
     full pytest discovery.
   - You want a documented "this MVP shipped" proof (the
     `verify_mvpNN.py` scripts are milestone markers).
2. **Future home for one-off tools.** Anything ad-hoc — debugging
   probes, calibration helpers, log parsers — lands here.

## The verifiers

| Script | Scope | Run when |
|---|---|---|
| `verify.py` | Vision filters, magnifier, image saver — pure unit | Smoke check after touching `processing/` or `storage/` |
| `verify_app.py` | App controller, state machine, event handlers | After touching `core/` |
| `verify_controls.py` | `MockControls` (keyboard) end-to-end | After touching keyboard fallback |
| `verify_integration.py` | Full app run, scripted events, headless | After cross-cutting changes |
| `verify_mvp02.py` | MVP 0.2 milestone (Pi camera, fullscreen, autofocus) | Historical reference |
| `verify_mvp03.py` | MVP 0.3 milestone (TCA6416A, MCP3221, GPIOControls) | Historical reference |

Run from the project root:

```bash
python3 scripts/dev/verify_mvp03.py
```

Or all at once:

```bash
for f in scripts/dev/verify*.py; do
  python3 "$f" 2>&1 | tail -1 | xargs -I{} printf "%-30s %s\n" "$f" "{}"
done
```

## When to prefer pytest

For day-to-day work on the CM5 venv (which has pytest installed), prefer:

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

It collects 265+ tests, parametrises cleanly, and is the authoritative
suite. The verifiers cover the same code paths but with less coverage and
no parametrisation. They exist because they were the *first* way the
codebase was tested back when the sandbox lacked pytest — keeping them
around costs nothing and the milestone-marker role is genuinely useful.