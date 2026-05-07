#!/usr/bin/env python
"""
Fairness Evaluation CLI - Easy batch generation and processing

Usage:
    python scripts/fairness_cli.py list-thresholds          # Show threshold options
    python scripts/fairness_cli.py generate --batch fairness_v1 --threshold 0.48 --providers openai grok
    python scripts/fairness_cli.py extract --batch fairness_v1 --extractors openai grok
    python scripts/fairness_cli.py metrics --batch fairness_v1
    python scripts/fairness_cli.py status --batch fairness_v1
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fairness_eval.fairness_evaluation import (
    get_negatively_predicted_instances,
    adjust_classification_threshold,
    get_attribute_combinations,
    generate_fairness_narratives,
    extract_from_fairness_narratives,
    compute_metrics_by_attribute_combinations,
    NARRATIVE_PROVIDERS,
    EXTRACTOR_PROVIDERS
)


def cmd_list_thresholds(args):
    """Show how many instances at different thresholds."""
    print("\n📊 THRESHOLD EXPLORATION")
    print("=" * 80)
    
    thresholds = [0.40, 0.42, 0.45, 0.48, 0.50, 0.52, 0.55]
    print(f"{'Threshold':<12} {'Instances':<12} {'Best for'}")
    print("-" * 80)
    
    for thresh in thresholds:
        count, _ = adjust_classification_threshold(args.dataset, thresh)
        best = " ← ~50 target" if 45 <= count <= 55 else ""
        print(f"{thresh:<12.2f} {count:<12d} {best}")
    
    print("\n💡 Choose threshold that gives you ~50 instances")


def cmd_generate(args):
    """Generate fairness narratives."""
    print(f"\n🚀 GENERATING FAIRNESS NARRATIVES")
    print(f"Batch: {args.batch}")
    print(f"Threshold: {args.threshold}")
    print(f"Providers: {', '.join(args.providers)}")
    
    # Get instances
    instances = get_negatively_predicted_instances(args.dataset, args.threshold)
    print(f"Instances: {len(instances)}")
    
    # Get combinations
    combinations = get_attribute_combinations(args.dataset)
    print(f"Combinations: {len(combinations)}")
    
    total = len(instances) * len(args.providers) * len(combinations)
    print(f"Total narratives: {total:,}")
    
    if not args.yes:
        response = input("\nProceed? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Generate
    stats = generate_fairness_narratives(
        dataset_name=args.dataset,
        instance_indices=instances,
        providers=args.providers,
        batch_name=args.batch,
        dry_run=args.dry_run
    )
    
    print(f"\n✅ Generation complete!")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")


def cmd_extract(args):
    """Extract from fairness narratives."""
    print(f"\n🚀 EXTRACTING FROM NARRATIVES")
    print(f"Batch: {args.batch}")
    print(f"Extractors: {', '.join(args.extractors)}")
    
    if not args.yes:
        response = input("\nProceed? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Extract
    stats = extract_from_fairness_narratives(
        dataset_name=args.dataset,
        extractors=args.extractors,
        batch_name=args.batch,
        dry_run=args.dry_run
    )
    
    print(f"\n✅ Extraction complete!")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")


def cmd_metrics(args):
    """Compute metrics for batch."""
    print(f"\n📊 COMPUTING METRICS")
    print(f"Batch: {args.batch}")
    
    metrics_df = compute_metrics_by_attribute_combinations(
        dataset_name=args.dataset,
        batch_name=args.batch
    )
    
    if not metrics_df.empty:
        # Save to CSV for reference
        csv_path = f"results/fairness_eval/{args.batch}/metrics_summary.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"\n✅ Metrics saved to: {csv_path}")


def cmd_status(args):
    """Check status of batch."""
    print(f"\n📋 BATCH STATUS: {args.batch}")
    print("=" * 80)
    
    batch_dir = Path(f"results/fairness_eval/{args.batch}")
    
    if not batch_dir.exists():
        print(f"❌ Batch not found: {batch_dir}")
        return
    
    # Count narratives
    narratives_dir = batch_dir / "narratives"
    if narratives_dir.exists():
        narrative_files = list(narratives_dir.glob("*/*.json"))
        print(f"\n📝 Narratives: {len(narrative_files)} files generated")
        
        # Count by provider
        providers = {}
        for f in narrative_files:
            provider = f.parent.name
            providers[provider] = providers.get(provider, 0) + 1
        
        print("   By provider:")
        for provider, count in sorted(providers.items()):
            print(f"     • {provider}: {count}")
    
    # Count extractions
    extractions_dir = batch_dir / "extractions"
    if extractions_dir.exists():
        extraction_files = list(extractions_dir.glob("*/*/*.json"))
        print(f"\n🔍 Extractions: {len(extraction_files)} files generated")
        
        # Count by extractor
        extractors = {}
        for f in extraction_files:
            extractor = f.parent.parent.name
            extractors[extractor] = extractors.get(extractor, 0) + 1
        
        print("   By extractor:")
        for extractor, count in sorted(extractors.items()):
            print(f"     • {extractor}: {count}")
    
    # Check metrics
    metrics_file = batch_dir / "metrics_by_batch.csv"
    if metrics_file.exists():
        print(f"\n📊 Metrics: ✅ metrics_by_batch.csv exists")
    else:
        print(f"\n📊 Metrics: ⏳ Not yet computed")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Fairness Evaluation CLI")
    parser.add_argument("--dataset", default="credit", choices=["credit", "law"],
                        help="Dataset to use")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # list-thresholds
    subparsers.add_parser("list-thresholds", help="List threshold options")
    
    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate narratives")
    gen_parser.add_argument("--batch", required=True, help="Batch name")
    gen_parser.add_argument("--threshold", type=float, required=True, help="Classification threshold")
    gen_parser.add_argument("--providers", nargs="+", default=NARRATIVE_PROVIDERS, help="LLM providers")
    gen_parser.add_argument("--dry-run", action="store_true", help="Don't actually generate")
    gen_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # extract
    ext_parser = subparsers.add_parser("extract", help="Extract from narratives")
    ext_parser.add_argument("--batch", required=True, help="Batch name")
    ext_parser.add_argument("--extractors", nargs="+", default=EXTRACTOR_PROVIDERS, help="Extractors to use")
    ext_parser.add_argument("--dry-run", action="store_true", help="Don't actually extract")
    ext_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    
    # metrics
    met_parser = subparsers.add_parser("metrics", help="Compute metrics")
    met_parser.add_argument("--batch", required=True, help="Batch name")
    
    # status
    stat_parser = subparsers.add_parser("status", help="Check batch status")
    stat_parser.add_argument("--batch", required=True, help="Batch name")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Route to command handler
    if args.command == "list-thresholds":
        cmd_list_thresholds(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "metrics":
        cmd_metrics(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
