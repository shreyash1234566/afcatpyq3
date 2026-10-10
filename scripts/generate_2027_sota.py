"""Single production entry point for the January 2027 forecast and practice paper.

Builds the shared integer blueprint, selects answered blueprint-matched practice
questions, validates all counts, and compiles the dashboard. No external LLM or
unsafe numeric-mutation fallback is used. Forecast weights remain provisional.
"""
import argparse
from build_unified_2027 import main as build_unified
from compile_dashboard_2027 import main as compile_dashboard


def main(use_2026_anchor=True, anchor_cycle='august'):
    if anchor_cycle != 'august':
        raise ValueError('January 2027 production uses the previous August anchor.')
    build_unified()
    compile_dashboard()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regime-2026', action='store_true', help='Compatibility flag; August 2026 is always the production anchor.')
    parser.add_argument('--anchor-cycle', choices=['august'], default='august')
    args = parser.parse_args()
    main(anchor_cycle=args.anchor_cycle)
