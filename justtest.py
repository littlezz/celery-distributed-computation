from common.decorator import set_debug
import logging
# logging.basicConfig(level=logging.DEBUG)
l = logging.getLogger(__name__)
l.setLevel(logging.DEBUG)
def gg():
    print('1')
    l.warn('ddddde')

# gg(debug=1)
gg()
print(logging)