import asyncio
import aiohttp
from aiohttp import web
from common import state
import json



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


async def node_status(request):
    "send node status to front-end"

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # async for msg in ws:
    #     if msg.tp == aiohttp.MsgType.text:
    #         status = msg.data
    #
    #     elif msg.tp == aiohttp.MsgType.error:
    #         status = state.node.OFFLINE
    #         ws_manager.pop(host)
    while True:
        try:
            ws.send_str(json.dumps(request.app['node_status']))
        except RuntimeError:
            break
        await asyncio.sleep(1)



