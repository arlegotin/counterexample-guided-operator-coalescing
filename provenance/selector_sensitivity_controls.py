#!/usr/bin/env python3
"""Owned end-to-end selector sensitivity; no pretrained/protected inference."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from poc.selector_controls import run_owned_selector_suite


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source-binding',type=Path)
    parser.add_argument('--source-binding-sha256')
    args=parser.parse_args(argv)
    if (args.source_binding is None)!=(args.source_binding_sha256 is None):parser.error('source binding path and SHA256 must be supplied together')
    binding=None if args.source_binding is None else {'path':str(args.source_binding),'sha256':args.source_binding_sha256}
    study='selector-tiny-bank-controls-v1' if binding is None else 'selector-tiny-bank-controls-integrated-v2'
    if args.out.resolve()!=ROOT/'artifacts'/study:
        raise ValueError('isolated owned study output must match the source-binding version: '+study)
    print(json.dumps(run_owned_selector_suite(args.out,**({} if binding is None else {'source_binding':binding})),sort_keys=True))


if __name__=='__main__':main()
