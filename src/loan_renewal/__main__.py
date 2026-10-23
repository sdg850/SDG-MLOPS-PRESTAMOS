"""Unified commands for the Cookiecutter Data Science project."""
from __future__ import annotations
import argparse
import json
import logging
from .config import DATA_PATH, MODELS_DIR, REPORTS_DIR


def main(argv=None):
    parser = argparse.ArgumentParser(description="Loan renewal: reproducible MLOps")
    parser.add_argument('--verbose', action='store_true')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'pipeline'):
        command = commands.add_parser(name)
        command.add_argument('--data', default=str(DATA_PATH))
        command.add_argument('--output', default=str(MODELS_DIR))
        command.add_argument('--reports', default=str(REPORTS_DIR / 'training'))
        command.add_argument('--tune', action='store_true')
        command.add_argument('--search-method', choices=['grid', 'random'], default='grid')
        command.add_argument('--track', action='store_true')
        if name == 'train':
            command.add_argument('--seed', type=int, default=42)
            command.add_argument('--no-cluster', action='store_true')
            command.add_argument('--smote', action='store_true')
            command.add_argument('--xgboost', action='store_true')
    command = commands.add_parser('predict')
    command.add_argument('--model', default=str(MODELS_DIR / 'model.joblib'))
    command.add_argument('--input', required=True)
    command.add_argument('--output', required=True)
    for name in ('version-data', 'check-env', 'experiments'):
        command = commands.add_parser(name, add_help=False)
        command.add_argument('arguments', nargs=argparse.REMAINDER)
    args, extra = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    if args.command in {'version-data', 'check-env', 'experiments'}:
        # Parse delegated arguments from the original tokens so --help reaches its parser.
        import sys
        tokens = list(argv if argv is not None else sys.argv[1:])
        delegated = tokens[tokens.index(args.command) + 1:]
        if args.command == 'version-data':
            from .versioning import main as run
            return run(delegated)
        if args.command == 'check-env':
            from .environment import main as run
            return run(delegated)
        choices = ['base', 'grid', 'random', 'compare', 'best', 'registry', 'all']
        action_parser = argparse.ArgumentParser(description='Tracked experiments and candidate Registry')
        action_parser.add_argument('action', choices=choices)
        action, rest = action_parser.parse_known_args(delegated)
        from .course import main as run
        run(action.action, rest)
        return 0
    if extra:
        parser.error('unrecognized arguments: ' + ' '.join(extra))
    if args.command == 'pipeline':
        from .orchestration import run_stages
        result = run_stages(args.data, args.output, tune=args.tune,
                            search_method=args.search_method, track=args.track, report_dir=args.reports)
    elif args.command == 'train':
        from .pipeline import train
        result = train(args.data, args.output, args.seed, not args.no_cluster, args.smote,
                       args.xgboost, args.tune, args.search_method, report_dir=args.reports)
        if args.track:
            from .tracking import track_report
            from .config import tracking_uri
            result['tracking'] = track_report(result, args.reports, tracking_uri=tracking_uri())
        result = {k: result[k] for k in ('winner', 'threshold', 'test', 'model_path', 'reload_prediction_parity')}
    else:
        from .predict import predict_file
        result = {'rows': len(predict_file(args.model, args.input, args.output)), 'output': args.output}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
