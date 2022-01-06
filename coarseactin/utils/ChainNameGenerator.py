import string
import itertools
import warnings

def chain_name_generator(format='cif'):
    allowed_characters = string.ascii_uppercase + string.ascii_lowercase + '0123456789'
    r = 0
    while True:
        r += 1
        if r == 2 and format == 'pdb':
            warnings.warn('The number of chains has exceeded the maximum allowable in the pdb format')
            break
        for a in itertools.product(allowed_characters, repeat=r):
            yield ''.join(a)
    while True:
        yield 'A'


if __name__ == '__main__':
    c = chain_name_generator(format='pdb')
    chains = [next(c) for _ in range(62**2)]
    assert chains[0] == 'A'
    assert chains[3] == 'D'
    assert chains[62]=='A'

    c = chain_name_generator()
    chains = [next(c) for _ in range(62**3+62**2+62+1)]
    assert chains[0]=='A'
    assert chains[62]=='AA'
    assert chains[62**2+62] == 'AAA'
    assert chains[62**3+62**2+62] == 'AAAA'
