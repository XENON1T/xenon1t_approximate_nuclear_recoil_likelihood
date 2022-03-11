
"""Console script for xepmts."""
import sys
import os

import click
from xe_likelihood.demo import demonstration


@click.group()
def main():
    """Console script for xepmts."""
    return 0

@main.command()
@click.option('--experiment', default='xenonnt',
            type=click.Choice(['xenonnt', 'xenon1t']),
            help='Experiment to display.')
@click.option('--show', is_flag=True, help='Plot results.')
@click.option('--savefig', is_flag=True, help='Save the results plot.')
@click.option('--path', default='', help='path to save results.')
def demo(experiment, show, savefig, path):
    if path:
        savefig = path
    return demonstration(experiment, show=show, savefig=savefig)

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
