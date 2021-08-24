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
    '@@@e    d @@',
    '@@@       @@',
    '@@@f      @@',
    '@@@    a  @@',
    '@@@  c   b@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@',
    '@@@@@@@@@@@@']

GOALS_LIST = [i for i in 'abcdef']
GOAL_GROUPS = {
    'G_1': {
        'goals': {
            'a': {
                'obtainable': False,
                'taken': False
            },
            'b': {
                'obtainable': False,
                'taken': False
            },
            'c': {
                'obtainable': True,
                'taken': False
            },
        },
        'urgent': False,
        'activated': False
    },
    'G_2': {
        'goals': {
            'd': {
                'obtainable': False,
                'taken': False
            },
            'e': {
                'obtainable': False,
                'taken': False
            },
            'f': {
                'obtainable': True,
                'taken': False,
            },
        },
        'urgent': False,
        'activated': False
    }
}

