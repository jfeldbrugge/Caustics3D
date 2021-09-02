from dataclasses import dataclass
from typing import Union
from functools import reduce
from operator import and_, xor, or_
from copy import copy
from enum import Enum
import sys

class Shape(Enum):
    EMPTY = 0
    SINGLE = 1
    WEDGE = 2
    CHOPSTICKS = 3
    TRIPOD = 4
    TRIANGLE = 5
    ZIGZAG = 6
    RING = 7
    SUNDIAL = 8
    FLAP = 9
    TOTAL = 10


@dataclass
class TetCase:
    variant: Shape
    vertices: list[Union[int, list[int]]]

    @property
    def flat_verts(self):
        return sum((i if isinstance(i, list) else [i] for i in self.vertices), [])

    @property
    def vertex_args(self):
        if not self.vertices:
            return ""
        return tuple(tuple(x) if isinstance(x, list) else x for x in self.vertices)

    def rust_case(self):
        return f"EdgeCase::{self.variant.name.title()}{self.vertex_args}"


tet_edges: list[set[int]] = [
    {i, j} for i in range(4) for j in range(i+1, 4)
]

def mask_to_edges(i: int) -> list[set[int]]:
    return [copy(e) for j, e in enumerate(tet_edges)
                    if i & (1 << j) > 0]

def marching_tet_case(i: int) -> TetCase:
    if i == 0: return TetCase(Shape.EMPTY, [])
    edges = mask_to_edges(i)
    verts = set(range(4))
    if len(edges) == 1:
        rest = list(verts - edges[0])
        return TetCase(Shape.SINGLE, [list(edges[0])] + rest)
    if len(edges) == 2:
        common = edges[0] & edges[1]
        if common:
            wedge = [s.pop() for s in [edges[0] - common, common, edges[1] - common]]
            return TetCase(Shape.WEDGE, [wedge, (verts - set(wedge)).pop()])
        else:
            return TetCase(Shape.CHOPSTICKS, [list(edges[0]), list(edges[1])])
    if len(edges) == 3:
        common = reduce(and_, edges)
        if common:
            ends = list(verts - common)
            return TetCase(Shape.TRIPOD, [common.pop(), ends])
        bound = reduce(xor, edges)
        if not bound:
            triangle_verts = edges[0] | edges[1] | edges[2]
            return TetCase(Shape.TRIANGLE, [list(triangle_verts), (verts - triangle_verts).pop()])
        v0 = bound.pop()
        v3 = bound.pop()
        v1 = next(s - {v0} for s in edges if v0 in s).pop()
        v2 = next(s - {v1} for s in edges if v1 in s and not v0 in s).pop()
        return TetCase(Shape.ZIGZAG, [v0, v1, v2, v3])
    if len(edges) == 4:
        bound = reduce(xor, edges)
        if not bound:
            v0 = edges[0].pop()
            v1 = edges[0].pop()
            v2 = next(s - {v1} for s in edges[1:] if v1 in s).pop()
            v3 = next(s - {v2} for s in edges if v2 in s and not v1 in s).pop()
            return TetCase(Shape.RING, [v0, v1, v2, v3])
        else:
            inverse = marching_tet_case(0x3F ^ i)
            return TetCase(Shape.SUNDIAL, inverse.vertices)
    if len(edges) == 5:
        inverse = marching_tet_case(0x3F ^ i)
        return TetCase(Shape.FLAP, inverse.vertices)
    if len(edges) == 6:
        return TetCase(Shape.TOTAL, [])

    raise ValueError("Unknown shape from encoding")

if __name__ == "__main__":
    cases = list(map(marching_tet_case, range(64)))

    expected_hist = {
        Shape.EMPTY: 1,
        Shape.SINGLE: 6,
        Shape.WEDGE: 12,
        Shape.CHOPSTICKS: 3,
        Shape.TRIPOD: 4,
        Shape.ZIGZAG: 12,
        Shape.TRIANGLE: 4,
        Shape.RING: 3,
        Shape.SUNDIAL: 12,
        Shape.FLAP: 6,
        Shape.TOTAL: 1
    }
    hist = {s: len([c for c in cases if c.variant == s])
            for s in Shape}

    assert all(sorted(c.flat_verts) == list(range(4))
               for c in cases
               if c.variant not in {Shape.EMPTY, Shape.TOTAL})
    print("All non-trivial variants mention four vertices", file=sys.stderr)
    assert expected_hist == hist
    print("Shape count is correct.", file=sys.stderr)
    print("-----------------------", file=sys.stderr)
    for s, i in hist.items():
        print(f"{s.name.title():16}", f"{i:2}", file=sys.stderr)
    print("-----------------------", file=sys.stderr)

    for i, c in enumerate(cases):
        print(f"0x{i:02X} => {c.rust_case()},")

