import asyncio
import aiohttp
from aiohttp import client
import psutil

from common.decorator import set_debug
from .settings import *
import logging
logger = logging.getLogger(__name__)


async def send_status(ws):
    while True:
        cpu_percent = str(psutil.cpu_percent())
        ws.send_str(cpu_percent)
        await asyncio.sleep(1)


async def receive_command(ws):
    async for msg in ws:
        if msg.tp == aiohttp.MsgType.text:
            # TODO:
            if msg.data == 'close cmd':
                await ws.close()
                break
            else:
                ws.send_str(msg.data + '/answer')
        elif msg.tp == aiohttp.MsgType.closed:
            break
        elif msg.tp == aiohttp.MsgType.error:
            break


async def communicate():
    session = client.ClientSession()
    ws  = await session.ws_connect(COORDINATOR_SERVER_URL)
    asyncio.ensure_future(send_status(ws))
    asyncio.ensure_future(receive_command(ws))




@set_debug
def run():
    logger.debug('debug mode!')
    loop = asyncio.get_event_loop()
    loop.create_task(communicate())
    loop.run_forever()


if __name__ == '__main__':
    run()