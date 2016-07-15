import asyncio
import aiohttp
from aiohttp import web
from common import state
import json
import psutil
import aiohttp_jinja2
import jinja2
from common.decorator import set_debug
from .setttings import *
import logging
logger = logging.getLogger('coordinator')

LOCALHOST = 'localhost'


async def receive_node_status(request):
    "receive node status from node"

    peername = request.transport.get_extra_info('peername')
    if peername is not None:
        host, port = peername
    else:
        raise web.HTTPBadRequest

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.info('node %s is online', host)

    ws_manager = request.app['node_ws_manager']
    ws_manager.update({host:ws})
    node_status = request.app['node_status']


    async for msg in ws:
        status = False
        if msg.tp == aiohttp.MsgType.text:
            status = msg.data
            # ws.send_str('server got your msg')

        elif msg.tp == aiohttp.MsgType.error:
            status = state.node.OFFLINE
            ws_manager.pop(host)

        if status is not False:
            node_status.update({host:status})

    logger.debug('Connect gone, mark as offline')
    node_status.update({host: state.node.OFFLINE})
    return ws


async def ws_node_status(request):
    "send node status to front-end"

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    logger.debug('Got a connect')
    ws.send_str(json.dumps(request.app['node_status']))

    try:
        async for m in ws:
            await asyncio.sleep(1)
            ws.send_str(json.dumps(request.app['node_status']))
        logger.debug('finish connect')

    except RuntimeError:
        logger.debug('ws lose connect')
    logger.debug('exit a connect')


@aiohttp_jinja2.template('welcome.jinja2')
async def welcome(request):
    # TODO:
    pass


async def start_task(request):
    pass


async def update_coordinator_cpu_info(node_status):
    while True:
        cpu_info = psutil.cpu_percent()
        node_status.update({LOCALHOST:cpu_info})
        await asyncio.sleep(1)




def app_update_router(app):
    app.router.add_route('GET', '/', welcome)
    app.router.add_route('GET', '/start', start_task)
    app.router.add_route('GET', '/ws_receive_node_status', receive_node_status)
    app.router.add_route('GET', '/ws_node_status', ws_node_status)
    app.router.add_static('/', 'coordinator/templates')


def on_shutdown(app):
    long_run_tasks = app['long_run_tasks']
    for task in long_run_tasks:
        task.cancel()
    logger.debug('cancel long run tasks')

    # TODO: cancel all celery task on node


def init_app():
    app = web.Application()
    app_update_router(app)

    # init
    node_status = dict()
    node_status.update((host, state.node.OFFLINE) for host in STATIC_NODE_HOSTS)
    app['node_status'] = node_status
    app['node_ws_manager'] = dict()

    long_run_tasks = list()
    long_run_tasks.append(app.loop.create_task(update_coordinator_cpu_info(node_status)))
    app['long_run_tasks'] = long_run_tasks

    aiohttp_jinja2.setup(app, loader=jinja2.PackageLoader('coordinator', 'templates'))

    app.on_shutdown.append(on_shutdown)
    return app


@set_debug
def run_server():
    logger.debug('debug mode start')
    app = init_app()
    web.run_app(app, shutdown_timeout=1)


if __name__ == '__main__':
    run_server()