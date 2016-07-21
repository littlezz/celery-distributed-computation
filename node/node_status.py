import asyncio
import aiohttp
from aiohttp import client
import psutil

from common.decorator import set_debug
from .settings import *
import logging
logger = logging.getLogger('node')


async def send_status(ws):
    while True:
        cpu_percent = str(psutil.cpu_percent())
        try:
            ws.send_str(cpu_percent)
            logger.debug('send cpu status, %s', cpu_percent)
        except RuntimeError:
            logger.info('server shutdown, return')
            return
        await asyncio.sleep(PERIOD_UPDATE)


# reference http://aiohttp.readthedocs.io/en/stable/client.html#websockets
# Which says that Websocket client can only use websocket task for both reading and writing
async def communicate():
    session = client.ClientSession()
    ws  = await session.ws_connect(COORDINATOR_SERVER_URL)
    logger.debug('ws connect')
    ws.send_str('client ok, connect')
    write_task = asyncio.ensure_future(send_status(ws))

    async for msg in ws:
        if msg.tp == aiohttp.MsgType.text:
            # TODO:
            if msg.data == 'close cmd':
                await ws.close()
                break
            else:
                logger.debug('recieve command %s', msg.data)
        elif msg.tp == aiohttp.MsgType.closed:
            break
        elif msg.tp == aiohttp.MsgType.error:
            break
    write_task.cancel()
    session.close()




@set_debug
def run():
    logger.debug('debug mode!')
    loop = asyncio.get_event_loop()
    loop.create_task(communicate())
    loop.run_forever()


if __name__ == '__main__':
    run()