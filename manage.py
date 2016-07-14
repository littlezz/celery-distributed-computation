from node import node_status
from coordinator.server import run_server
import click
import logging
logging.basicConfig()

@click.group(invoke_without_command=True)
def manage():
    pass


@manage.command()
@click.option('--debug', is_flag=True)
def node(debug):
    click.echo('node start!')
    node_status.run(debug=debug)


@manage.command()
@click.option('--debug', is_flag=True)
def server(debug):
    click.echo('server start!')
    run_server(debug=debug)


if __name__ == '__main__':
    manage()