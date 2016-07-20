import random
from string import ascii_letters


def get_random_string(bit=32):
    return ''.join(random.choice(ascii_letters))