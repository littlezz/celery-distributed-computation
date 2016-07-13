import asyncio
import aiohttp
from aiohttp import web
from common import state
import json
import psutil
from .setttings import *


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

    ws_manager = request.app['node_ws_manager']
    ws_manager.update({host:ws})
    node_status = request.app['node_status']

    async for msg in ws:
        status = False
        if msg.tp == aiohttp.MsgType.text:
            status = msg.data

        elif msg.tp == aiohttp.MsgType.error:
            status = state.node.OFFLINE
            ws_manager.pop(host)

        if status is not False:
            node_status.update({host:status})


async def ws_node_status(request):
    "send node status to front-end"

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    while True:
        try:
            ws.send_str(json.dumps(request.app['node_status']))
        except RuntimeError:
            break
        await asyncio.sleep(1)


async def welcome(request):
    pass


async def update_coordinator_cpu_info(node_status):
    while True:
        cpu_info = psutil.cpu_percent()
        node_status.update({LOCALHOST:cpu_info})
        await asyncio.sleep(1)




def app_update_router(app):
    app.router.add_route('GET', '/', welcome)
    app.router.add_route('GET', '/ws_receive_node_status', receive_node_status)
    app.router.add_route('GET', '/ws_node_status', ws_node_status)


def init_app():
    app = web.Application()
    app_update_router(app)

    # init
    node_status = dict()
    node_status.update((host, state.node.OFFLINE) for host in STATIC_NODE_HOSTS)
    app['node_status'] = node_status
    app['node_ws_manager'] = dict()

    app.loop.create_task(update_coordinator_cpu_info(node_status))
    return app


def run_server():
    app = init_app()
    web.run_app(app)


if __name__ == '__main__':
    run_server()