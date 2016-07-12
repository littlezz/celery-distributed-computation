from node import node_status
import click


@click.group(invoke_without_command=True)
def manage():
    pass


@manage.command()
def node():
    click.echo('node start!')
    node_status.run()


if __name__ == '__main__':
    manage()