# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'A' means apply spawn point
# '' is empty space

HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

CLEANUP_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH      BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR  P    BBBB@',
    '@RRRRR    P BBBBB@',
    '@HHHHH       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR   P P BBBB@',
    '@HHHHH   P  BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH P   BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH    P  BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH  P P BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@']

WARD_MAP_1 = [
    '@@@@@@@@@@@',
    '@ B  B   @@',
    '@        @@',
    '@ B  B   @@',
    '@@@@@@   @@',
    '@ B  B   @@',
    '@       S@@',
    '@ B  B   @@',
    '@@@@@@   @@',
    '@ B  B   @@',
    '@       T@@',
    '@ B  B   @@',
    '@@@@@@@@@@@']
# WARD_MAP_6
WARD_MAP_6 = [
    '@@@@@@@@@@@@',
    '@a   b   c@@',
    '@         @@',
    '@d       e@@',
    '@@@@  @@@@@@',
    '@        T@@',
    '@         @@',
    '@    S    @@',
    '@  @@@@   @@',
    '@f   g@   @@',
    '@     @   @@',
    '@h   k@   @@',
    '@@@@@@@   @@',
    '@         @@',
    '@  n  m  l@@',
    '@@@@@@@@@@@@']

WARD_MAP = [
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@c     @@@',
    '@@@      @@@',
    '@@@a    b@@@',
    '@@@      @@@',
    '@@@d     @@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@']

WARD_MAP_2 = [
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@c S   @@@',
    '@@@      @@@',
    '@@@a b   @@@',
    '@@@      @@@',
    '@@@      @@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@']

WARD_MAP_V2_1 = [
    # 012345678901
    '@@@@@@@@@@@@',  # 0
    '@     d@  @@',  # 1
    '@ e    @  @@',  # 2
    '@        S@@',  # 3
    '@  f      @@',  # 4
    '@ @@@    @@@',  # 5
    '@  a     T@@',  # 6
    '@         @@',  # 7
    '@c        @@',  # 8
    '@   b     @@',  # 9
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@']

GOALS_LIST = {
    'a': {'location': None, 'urgency': False, 'requires': None},
    'b': {'location': None, 'urgency': False, 'requires': None},
    'c': {'location': None, 'urgency': False, 'requires': ['b','a']},
    'd': {'location': None, 'urgency': False, 'requires': ['e']},
    'e': {'location': None, 'urgency': False, 'requires': None},
    'f': {'location': None, 'urgency': False, 'requires': None},
    'S': {'location': None, 'urgency': True, 'requires': None},
    'T': {'location': None, 'urgency': False, 'requires': None}
}
