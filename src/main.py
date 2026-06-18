import os
import sys
import argparse
import json
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Ensure src directory is in PYTHONPATH
src_path = Path(__file__).resolve().parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from loguru import logger
from utils.config import get_app_config, config_instance
from utils.validators import validate_dataset_json
from core.evaluator import Evaluator, EvaluationRunSummary, TestCaseResult
from reporters.json_reporter import export_to_json
from reporters.html_reporter import generate_html_report
from reporters.dashboard import serve_dashboard

def cmd_init(args):
    """Scaffold a new project directory structure."""
    project_name = args.project_name
    target_dir = Path(os.getcwd()) / project_name
    
    logger.info(f"Scaffolding new AI evaluation project: {project_name}")
    
    # Subdirectories
    subdirs = [
        "config",
        "src/core",
        "src/tests/test_suites",
        "src/reporters",
        "src/data/datasets",
        "src/data/prompts",
        "reports"
    ]
    
    for sd in subdirs:
        (target_dir / sd).mkdir(parents=True, exist_ok=True)

    # We write a simple README and configuration skeleton
    logger.info("Writing basic project configuration skeletons...")
    
    # Write config/config.yaml placeholder
    with open(target_dir / "config" / "config.yaml", "w") as f:
        yaml.safe_dump({
            "project": {"name": project_name, "version": "1.0.0"},
            "llm": {"default_model": "mock"},
            "testing": {"parallel_workers": 4, "timeout": 30, "retry_attempts": 3},
            "reporting": {"output_dir": "./reports", "formats": ["html", "json"]},
            "datasets": {
                "golden_path": "./src/data/datasets/golden_dataset.json",
                "edge_path": "./src/data/datasets/edge_cases.json"
            }
        }, f)

    # Write a basic golden dataset
    with open(target_dir / "src" / "data" / "datasets" / "golden_dataset.json", "w") as f:
        json.dump({
            "test_cases": [
                {
                    "id": "TC001",
                    "category": "relevance",
                    "prompt": "What is the capital of France?",
                    "context": "France is a country in Europe. Paris is the capital.",
                    "expected_response": "Paris",
                    "expected_metrics": {"relevance_score": 0.8},
                    "domain": "geography",
                    "difficulty": "easy"
                }
            ]
        }, f, indent=2)

    logger.info(f"Project '{project_name}' initialized successfully.")

def cmd_validate(args):
    """Validate a dataset JSON schema."""
    dataset_file = args.dataset
    if not os.path.exists(dataset_file):
        # Check in default location if relative
        dataset_file = Path(config_instance.datasets.golden_path).parent / dataset_file
        if not dataset_file.exists():
            logger.error(f"Dataset file not found at {args.dataset}")
            sys.exit(1)

    logger.info(f"Validating dataset JSON format: {dataset_file}")
    is_valid, errors = validate_dataset_json(str(dataset_file))
    if is_valid:
        logger.info("Dataset is VALID!")
    else:
        logger.error("Dataset has validation errors:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

def cmd_run(args):
    """Execute an evaluation run."""
    suite = args.suite
    model = args.model or config_instance.llm.default_model

    # Map dataset path based on shortcuts
    dataset_path = None
    if suite == "golden":
        dataset_path = config_instance.datasets.golden_path
    elif suite == "edge":
        dataset_path = config_instance.datasets.edge_path
    elif suite == "domain":
        dataset_path = config_instance.datasets.domain_path
    elif suite == "all" or not suite:
        dataset_path = config_instance.datasets.golden_path  # Default to golden
    else:
        dataset_path = suite

    # Resolve paths relative to PROJECT_ROOT if needed
    p_path = Path(dataset_path)
    if not p_path.is_absolute():
        p_path = Path(__file__).resolve().parent.parent / dataset_path

    if not p_path.exists():
        logger.error(f"Dataset path not found: {p_path}")
        sys.exit(1)

    logger.info(f"Starting evaluation runner on dataset: {p_path} with model: {model}")
    
    # Initialize evaluator
    evaluator = Evaluator(config_instance)
    
    try:
        summary = evaluator.run_suite(str(p_path), model)
        
        # Output results
        from tabulate import tabulate
        
        print("\n" + "="*50)
        print("EVALUATION RUN COMPLETED")
        print("="*50)
        print(f"Run ID:        {summary.run_id}")
        print(f"Model:         {summary.model}")
        print(f"Pass Rate:     {summary.pass_rate * 100:.1f}%")
        print(f"Total Cases:   {summary.total_test_cases}")
        print(f"Passed:        {summary.passed_count}")
        print(f"Failed:        {summary.failed_count}")
        print(f"Total Tokens:  {summary.total_tokens}")
        print(f"Total Cost:    ${summary.total_cost:.6f}")
        print(f"Avg Latency:   {summary.avg_latency:.2f}s")
        print("="*50)
        
        # Metric averages table
        metric_table = [[k, f"{v:.3f}"] for k, v in summary.metric_averages.items()]
        print("\nMetric Averages:")
        print(tabulate(metric_table, headers=["Metric", "Average Score"], tablefmt="grid"))

        # Detailed test results summary table
        detailed_table = []
        for r in summary.results:
            status = "PASS" if r.passed else "FAIL"
            scores = ", ".join([f"{k.replace('_score', '')}:{v.score:.2f}" for k, v in r.metrics.items()])
            detailed_table.append([r.test_case_id, r.category, status, f"{r.latency:.2f}s", scores])
            
        print("\nTest Case Breakdown:")
        print(tabulate(detailed_table, headers=["ID", "Category", "Status", "Latency", "Scores"], tablefmt="simple"))

        # Export reports
        output_dir = config_instance.reporting.output_dir
        json_path = export_to_json(summary, output_dir)
        html_path = generate_html_report(summary, output_dir)
        
        print(f"\nSaved JSON Report: {json_path}")
        print(f"Saved HTML Dashboard: {html_path}\n")

    except Exception as e:
        logger.exception(f"Run aborted due to critical error: {str(e)}")
        sys.exit(1)

def cmd_report(args):
    """Generate reports from a previous JSON run log."""
    output_dir = Path(config_instance.reporting.output_dir)
    json_files = sorted(output_dir.glob("report_run_*.json"))
    
    if not json_files:
        logger.error(f"No previous evaluation run JSON logs found in {output_dir}")
        sys.exit(1)

    latest_json = json_files[-1]
    logger.info(f"Compiling HTML report from latest run log: {latest_json}")
    
    try:
        with open(latest_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Reconstruct Pydantic model
        summary = EvaluationRunSummary(**data)
        html_path = generate_html_report(summary, str(output_dir))
        print(f"Generated HTML Report: {html_path}")
    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}")
        sys.exit(1)

def cmd_compare(args):
    """Compare performance across two prompt versions."""
    v1_file = Path(args.v1)
    v2_file = Path(args.v2)
    model = args.model or config_instance.llm.default_model

    if not v1_file.exists() or not v2_file.exists():
        logger.error("Prompt templates YAML paths must exist.")
        sys.exit(1)

    logger.info(f"Comparing Prompts: V1 ({v1_file.name}) vs V2 ({v2_file.name}) using model '{model}'")

    # Load templates
    with open(v1_file, "r", encoding="utf-8") as f:
        prompts_v1 = yaml.safe_load(f).get("prompts", {})
    with open(v2_file, "r", encoding="utf-8") as f:
        prompts_v2 = yaml.safe_load(f).get("prompts", {})

    # Load default golden dataset
    dataset_path = Path(config_instance.datasets.golden_path)
    if not dataset_path.exists():
        dataset_path = Path(__file__).resolve().parent.parent / config_instance.datasets.golden_path
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    test_cases = dataset.get("test_cases", [])
    evaluator = Evaluator(config_instance)

    # Evaluate v1 templates
    logger.info("Evaluating V1 Prompts...")
    results_v1 = []
    for tc in test_cases:
        tc_cat = tc.get("domain", "default")
        p_v1 = prompts_v1.get(tc_cat, prompts_v1.get("default", {}))
        tc_clone = tc.copy()
        tc_clone["prompt"] = p_v1.get("user", "{prompt}").format(prompt=tc.get("prompt"), context=tc.get("context", ""))
        res = evaluator.evaluate_test_case(tc_clone, model)
        results_v1.append(res)

    # Evaluate v2 templates
    logger.info("Evaluating V2 Prompts...")
    results_v2 = []
    for tc in test_cases:
        tc_cat = tc.get("domain", "default")
        p_v2 = prompts_v2.get(tc_cat, prompts_v2.get("default", {}))
        tc_clone = tc.copy()
        tc_clone["prompt"] = p_v2.get("user", "{prompt}").format(prompt=tc.get("prompt"), context=tc.get("context", ""))
        res = evaluator.evaluate_test_case(tc_clone, model)
        results_v2.append(res)

    # Display comparison
    from tabulate import tabulate
    comparison_table = []
    for r1, r2 in zip(results_v1, results_v2):
        # Fetch relevance/hallucination average score changes
        v1_avg = sum(m.score for m in r1.metrics.values()) / len(r1.metrics) if r1.metrics else 0.0
        v2_avg = sum(m.score for m in r2.metrics.values()) / len(r2.metrics) if r2.metrics else 0.0
        diff = v2_avg - v1_avg
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        
        comparison_table.append([
            r1.test_case_id,
            r1.category,
            f"{v1_avg:.2f}",
            f"{v2_avg:.2f}",
            diff_str,
            "PASS" if r2.passed else "FAIL"
        ])

    print("\nPrompt Regression Comparison Table:")
    print(tabulate(comparison_table, headers=["Case ID", "Category", "V1 Score", "V2 Score", "Diff", "V2 Status"], tablefmt="grid"))

def cmd_analyze(args):
    """Analyze failure patterns from a previous run log."""
    run_id = args.run_id
    pattern_type = args.pattern

    output_dir = Path(config_instance.reporting.output_dir)
    
    if run_id == "latest" or not run_id:
        json_files = sorted(output_dir.glob("report_run_*.json"))
        if not json_files:
            logger.error("No previous runs found to analyze.")
            sys.exit(1)
        run_file = json_files[-1]
    else:
        run_file = output_dir / f"report_run_{run_id}.json"
        if not run_file.exists():
            run_file = output_dir / f"report_{run_id}.json"
            if not run_file.exists():
                logger.error(f"Run report file not found for run-id: {run_id}")
                sys.exit(1)

    logger.info(f"Analyzing run reports file: {run_file}")
    with open(run_file, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    results = run_data.get("results", [])
    failures = [r for r in results if not r.get("passed")]

    print("\n" + "="*50)
    print(f"FAILURE ANALYSIS SUMMARY (Total Failures: {len(failures)}/{len(results)})")
    print("="*50)

    if not failures:
        print("Success! No failed test cases detected in this run.")
        return

    # Pattern Detection
    if pattern_type == "detection" or pattern_type == "all":
        print("\n[Pattern Detection Details]")
        refusals = 0
        hallucinations = 0
        completeness_fails = 0
        
        for f in failures:
            resp = f.get("response_text", "").lower()
            # Safety refusal check
            if "cannot fulfill" in resp or "violates" in resp or "policy" in resp or "sorry" in resp:
                refusals += 1
                
            metrics = f.get("metrics", {})
            h_score = metrics.get("hallucination_score", {}).get("score", 1.0)
            if h_score < metrics.get("hallucination_score", {}).get("threshold", 0.8):
                hallucinations += 1
                
            comp_score = metrics.get("completeness_score", {}).get("score", 1.0)
            if comp_score < metrics.get("completeness_score", {}).get("threshold", 0.8):
                completeness_fails += 1
                
        print(f"  - Safety Policy Refusals:   {refusals}")
        print(f"  - Grounding/Hallucinations: {hallucinations}")
        print(f"  - Completeness/Length fails: {completeness_fails}")

    # Group failures by category
    category_fail = {}
    domain_fail = {}
    for f in failures:
        cat = f.get("category")
        category_fail[cat] = category_fail.get(cat, 0) + 1
        dom = f.get("domain")
        domain_fail[dom] = domain_fail.get(dom, 0) + 1

    from tabulate import tabulate
    print("\nFailures by Category:")
    print(tabulate(category_fail.items(), headers=["Category", "Fail Count"], tablefmt="simple"))

    print("\nFailures by Domain:")
    print(tabulate(domain_fail.items(), headers=["Domain", "Fail Count"], tablefmt="simple"))

    print("\nTop Failure Details:")
    for idx, f in enumerate(failures[:5]):
        print(f"\n{idx+1}. Test Case: {f.get('test_case_id')} | Category: {f.get('category')} | Domain: {f.get('domain')}")
        print(f"   Prompt: '{f.get('prompt')[:60]}...'")
        print("   Metric Failures:")
        for m_name, m_val in f.get("metrics", {}).items():
            if not m_val.get("passed"):
                print(f"     * {m_name}: Score {m_val.get('score')} < Threshold {m_val.get('threshold')} (Details: {m_val.get('details')})")

def cmd_dashboard(args):
    """Run the local dashboard server."""
    port = args.port
    output_dir = config_instance.reporting.output_dir
    logger.info("Initializing local report viewer...")
    serve_dashboard(directory=output_dir, port=port)

def cli():
    parser = argparse.ArgumentParser(description="AI Quality Evaluation & Prompt Regression CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. init command
    init_parser = subparsers.add_parser("init", help="Initialize a new evaluation project structure")
    init_parser.add_argument("--project-name", default="my_evaluation_project", help="Name of the project folder")

    # 2. validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a dataset JSON schema")
    validate_parser.add_argument("--dataset", required=True, help="Path to JSON dataset file to validate")

    # 3. run command
    run_parser = subparsers.add_parser("run", help="Run quality evaluations on a test dataset")
    run_parser.add_argument("--suite", default="golden", help="Test suite name (golden, edge, domain, all, or absolute path to custom JSON dataset)")
    run_parser.add_argument("--model", help="Override default model provider to evaluate")

    # 4. report command
    subparsers.add_parser("report", help="Compile HTML dashboard reports from the latest run logs")

    # 5. compare command
    compare_parser = subparsers.add_parser("compare", help="Compare scores across two prompt versions")
    compare_parser.add_argument("--v1", required=True, help="YAML prompt template version 1 file")
    compare_parser.add_argument("--v2", required=True, help="YAML prompt template version 2 file")
    compare_parser.add_argument("--model", help="LLM model provider to evaluate")

    # 6. analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Perform failure categorization and pattern analysis")
    analyze_parser.add_argument("--run-id", default="latest", help="Specific run ID to analyze")
    analyze_parser.add_argument("--pattern", choices=["detection", "all"], default="all", help="Analysis model pattern detection focus")

    # 7. dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Launch local dashboard web server")
    dashboard_parser.add_argument("--port", type=int, default=8000, help="Port to serve the dashboard")

    args = parser.parse_args()

    # Route command
    if args.command == "init":
        cmd_init(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)

if __name__ == "__main__":
    cli()
