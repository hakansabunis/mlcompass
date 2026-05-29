# Changelog

All notable changes to `ml-copilot` are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Vision
Full ML pipeline assistant: one CLI that follows your project from
data through training to deployment, with persistent project context.

### Roadmap

#### Faz 1 — v0.1 (in progress)
- `ml-copilot init <project>` — initialize project context
- `ml-copilot advise <data>` — dataset analysis + model + feature
  engineering recommendations + pitfall warnings
- `.mlcopilot/` directory with project metadata and persistent context
- Rich terminal UI with streaming agent output

#### Faz 2 — v0.2 (planned)
- `ml-copilot audit <script>` — static training-script analysis
- `ml-copilot watch <script>` — live training monitor with plateau,
  overfitting, NaN, and divergence detection
- `ml-copilot compare <run-a> <run-b>` — run-to-run diff with
  LLM-explained hypotheses
- Permission-gated config edits and training restarts
- TensorBoard / W&B / plain-text log support

#### Faz 3 — v0.3 (planned)
- `ml-copilot evaluate <results>` — post-training analysis
- Threshold optimization
- Confusion matrix interpretation
- Hard-example surfacing
- Fairness checks

#### Faz 4 — v0.4 (planned)
- `ml-copilot deploy --target <X>` — deployment readiness check
- Model size and inference latency estimation
- Dependency consistency verification
- ONNX / TorchScript conversion advice
- A/B testing and monitoring guidance

#### Beyond v1.0 (future)
- Jupyter magic (`%mlc`)
- Hosted version
- Multi-language support (R, Julia)
- Plugin system for custom advisors

## [0.0.1] — Unreleased

### Added
- Project scaffolding (`pyproject.toml`, MIT license, `.gitignore`)
- README with full pipeline vision and feature roadmap
- `ARCHITECTURE.md` — comprehensive design document covering project
  context format, agent hierarchy, tool catalog, permission strategy,
  module organization, CLI structure, data flow, and non-goals
- `agentlite-py` as the agent backbone dependency
- CLI entry point declaration (`ml-copilot` command, not yet implemented)
