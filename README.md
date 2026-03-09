Sentinel Data Agent

Sentinel is an agentic data analysis system that automatically inspects tabular datasets, builds an execution plan, runs exploratory analysis, evaluates modeling readiness, and produces a structured report.

The project demonstrates how to build a deterministic, reliable data-analysis agent using a capability-based architecture, persistent run state, and an orchestrated execution pipeline.

The goal of Sentinel is not just to analyze data, but to show how agentic systems can be engineered with transparency, reproducibility, and safety constraints.

PROJECT MOTIVATION

Many modern AI agents rely heavily on LLM reasoning loops. While powerful, those systems can be difficult to audit and reproduce.

Sentinel explores a different design philosophy.

The agent performs structured analysis using deterministic capabilities, stores every step as artifacts, and maintains an explicit execution state.

This approach provides several advantages.

The system is reproducible because every run produces artifacts and checkpoints.
The system is observable because the state machine exposes progress and results.
The system is reliable because unsafe modeling conditions are explicitly detected and blocked.

The project demonstrates how an agentic architecture can be built with strong engineering guarantees instead of opaque reasoning loops.

ARCHITECTURE OVERVIEW

Sentinel is organized into four main layers.

CLI Layer  
Exposes commands such as sentinel run, sentinel status, and sentinel report.

Orchestrator Layer  
Coordinates the execution of capabilities using a DAG-based execution plan.

Capabilities Layer  
Implements deterministic analysis steps such as schema detection, correlation analysis, and baseline modeling.

State Layer  
Persists run state, artifacts, and checkpoints.

Each dataset analysis is represented as a Run, which progresses through multiple phases:

Dataset profiling  
Schema analysis  
Target inference  
EDA summary  
Correlation analysis  
Feature readiness evaluation  
Baseline modeling  
Final report generation

ARCHITECTURE FLOW

User  
↓  
CLI (Typer)  
↓  
Bootstrap Orchestrator  
↓  
Execution Plan (DAG)  
↓  
EDA Summary  
↓  
Correlation Analysis  
↓  
Feature Readiness  
↓  
Baseline Classification  
↓  
Final Report  
↓  
Artifacts + Run State (JSON Store)

Every step updates the run state, persists artifacts, and allows the execution to be resumed or inspected.

AGENT DESIGN DECISIONS

Deterministic Capabilities

Instead of relying on LLM reasoning loops, Sentinel executes deterministic analysis functions. This ensures results are reproducible and easier to debug.

Capability-Based Architecture

Each analysis step is implemented as a separate capability module. This modular design allows new analysis steps to be added without modifying the orchestrator.

Explicit Execution Plan

Before running analysis, Sentinel builds a DAG-style execution plan. This makes the agent’s behavior predictable and inspectable.

Persistent Run State

Every run is stored with its status, artifacts, and checkpoints. This enables resumability, observability, and debugging of long-running analysis workflows.

Safety Constraints for Modeling

The system explicitly checks statistical conditions before training models. If the dataset violates basic assumptions, modeling is blocked instead of producing misleading metrics.

Artifact-Based Observability

Every capability writes structured artifacts. This creates a full trace of the agent’s reasoning and decisions.

EXECUTION PIPELINE

When a dataset is analyzed, Sentinel executes the following pipeline.

The system first profiles the dataset and evaluates its quality.
Then it infers a possible modeling target using heuristics.
After that it builds an execution plan describing which analysis steps should run.

The agent then executes capabilities sequentially.

Exploratory data analysis summarizes numeric and categorical columns.
Correlation analysis identifies relationships between numeric features.
Feature readiness evaluates whether the dataset is suitable for modeling.

If the dataset meets minimum requirements, a baseline classification model is trained.
If not, the agent safely blocks modeling and reports the reason.

Finally, the system produces a final structured report summarizing the analysis.

EXAMPLE RUN

Run Sentinel on a dataset:

sentinel run examples/example.csv

Example output:

Run created
run_id: sda_XXXXXXXX
dataset: examples/example.csv
rows: 4
columns: 3

eda summary: artifacts/.../eda_summary.json
correlation analysis: artifacts/.../correlation_analysis.json
feature readiness: artifacts/.../feature_readiness.json
baseline classification: artifacts/.../baseline_classification.json
final report: artifacts/.../final_report.json

Check run status:

sentinel status <run_id>

Generate a report summary:

sentinel report <run_id>

PROJECT STRUCTURE

sentinel-data-agent

src/sentinel
capabilities
schema_analysis.py
target_inference.py
eda_summary.py
correlation_analysis.py
feature_readiness.py
baseline_classification.py
final_report.py

orchestrator
bootstrap.py

state
store.py

cli
main.py

artifacts
examples
tests
README.md

TESTING

The project includes unit and integration tests validating each layer of the system.

Capabilities tests validate:
schema detection
target inference
exploratory analysis
correlation analysis
feature readiness
baseline modeling
final report generation

The orchestrator test validates the full pipeline execution.

The CLI test validates end-to-end execution through the command line interface.

Run the test suite:

pytest -v

SAFETY AND RELIABILITY DESIGN

Sentinel deliberately avoids unsafe modeling behavior.

The agent performs validation before training any model.

Examples of blocking conditions include datasets where the least populated class contains fewer than two samples.

When a blocking condition occurs, the agent does not attempt to train a model and instead reports the reason.

This design ensures the system avoids producing misleading results.

EXAMPLE ARTIFACT OUTPUT

artifacts/
sda_<run_id>/
dataset_profile.json
dataset_quality.json
schema_analysis.json
execution_plan.json
eda_summary.json
correlation_analysis.json
feature_readiness.json
baseline_classification.json
final_report.json

These artifacts allow the full analysis process to be audited and reproduced.

TECHNOLOGIES USED

Python  
Pandas  
Scikit-learn  
Typer CLI  
Pytest  
JSON artifact storage

FUTURE IMPROVEMENTS

Planned extensions include:

Support for regression pipelines  
Automated feature engineering  
Richer dataset quality diagnostics  
Visualization reports  
Optional LLM-based analysis layers

AUTHOR

Anna Carolina Oller de Castro

AI Engineering and Data Systems student  
Focused on Agentic AI, data pipelines, and reliable AI systems