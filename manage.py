from node import node_status
from coordinator import server
import click


@click.group(invoke_without_command=True)
def manage():
    pass


@manage.command()
def node():
    click.echo('node start!')
    node_status.run()


@manage.command()
def server():
    click.echo('server start!')
    server.run_server()


if __name__ == '__main__':
    manage()