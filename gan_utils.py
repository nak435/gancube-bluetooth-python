from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Import the GanCubeMove type - assuming it exists in a corresponding module
    from gan_cube_protocol import GanCubeMove

import time
from typing import List, Optional, Tuple

def now() -> int:
    """
    Return current host clock timestamp with millisecond precision
    Use monotonic clock when available
    @returns Current host clock timestamp in milliseconds
    """
    # Use perf_counter for high precision timing (equivalent to performance.now())
    return int(time.perf_counter() * 1000)

def linregress(X: List[Optional[float]], Y: List[Optional[float]]) -> Tuple[float, float]:
    sumX = 0
    sumY = 0
    sumXY = 0
    sumXX = 0
    sumYY = 0
    n = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        if x is None or y is None:
            continue
        n += 1
        sumX += x
        sumY += y
        sumXY += x * y
        sumXX += x * x
        sumYY += y * y
    varX = n * sumXX - sumX * sumX
    covXY = n * sumXY - sumX * sumY
    slope = 1 if varX < 1e-3 else covXY / varX
    intercept = 0 if n < 1 else sumY / n - slope * sumX / n
    return (slope, intercept)

def cubeTimestampLinearFit(cubeMoves: List[GanCubeMove]) -> List[GanCubeMove]:
    """
    Use linear regression to fit timestamps reported by cube hardware with host device timestamps
    @param cubeMoves List representing window of cube moves to operate on
    @returns New copy of move list with fitted cubeTimestamp values
    """
    res: List[GanCubeMove] = []
    # Calculate and fix timestamp values for missed and recovered cube moves.
    if len(cubeMoves) >= 2:
        # 1st pass - tail-to-head, align missed move cube timestamps to next move -50ms
        for i in range(len(cubeMoves) - 1, 0, -1):
            if cubeMoves[i].cubeTimestamp is not None and cubeMoves[i - 1].cubeTimestamp is None:
                cubeMoves[i - 1].cubeTimestamp = cubeMoves[i].cubeTimestamp - 50
        # 2nd pass - head-to-tail, align missed move cube timestamp to prev move +50ms
        for i in range(len(cubeMoves) - 1):
            if cubeMoves[i].cubeTimestamp is not None and cubeMoves[i + 1].cubeTimestamp is None:
                cubeMoves[i + 1].cubeTimestamp = cubeMoves[i].cubeTimestamp + 50
    # Apply linear regression to the cube timestamps
    if len(cubeMoves) > 0:
        slope, intercept = linregress([m.cubeTimestamp for m in cubeMoves], [m.localTimestamp for m in cubeMoves])
        first = round(slope * cubeMoves[0].cubeTimestamp + intercept)
        for m in cubeMoves:
            res.append(GanCubeMove(
                face=m.face,
                direction=m.direction,
                move=m.move,
                localTimestamp=m.localTimestamp,
                cubeTimestamp=round(slope * m.cubeTimestamp + intercept) - first
            ))
    return res

def cubeTimestampCalcSkew(cubeMoves: List[GanCubeMove]) -> float:
    """
    Calculate time skew degree in percent between cube hardware and host device
    @param cubeMoves List representing window of cube moves to operate on
    @returns Time skew value in percent
    """
    if not len(cubeMoves):
        return 0
    slope, _ = linregress([m.localTimestamp for m in cubeMoves], [m.cubeTimestamp for m in cubeMoves])
    return round((slope - 1) * 100000) / 1000

CORNER_FACELET_MAP = [
    [8, 9, 20],   # URF
    [6, 18, 38],  # UFL
    [0, 36, 47],  # ULB
    [2, 45, 11],  # UBR
    [29, 26, 15], # DFR
    [27, 44, 24], # DLF
    [33, 53, 42], # DBL
    [35, 17, 51]  # DRB
]

EDGE_FACELET_MAP = [
    [5, 10],  # UR
    [7, 19],  # UF
    [3, 37],  # UL
    [1, 46],  # UB
    [32, 16], # DR
    [28, 25], # DF
    [30, 43], # DL
    [34, 52], # DB
    [23, 12], # FR
    [21, 41], # FL
    [50, 39], # BL
    [48, 14]  # BR
]

def toKociembaFacelets(cp: List[int], co: List[int], ep: List[int], eo: List[int]) -> str:
    """

    Convert Corner/Edge Permutation/Orientation cube state to the Kociemba facelets representation string

    Example - solved state:
      cp = [0, 1, 2, 3, 4, 5, 6, 7]
      co = [0, 0, 0, 0, 0, 0, 0, 0]
      ep = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
      eo = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      facelets = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"
    Example - state after F R moves made:
      cp = [0, 5, 2, 1, 7, 4, 6, 3]
      co = [1, 2, 0, 2, 1, 1, 0, 2]
      ep = [1, 9, 2, 3, 11, 8, 6, 7, 4, 5, 10, 0]
      eo = [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
      facelets = "UUFUUFLLFUUURRRRRRFFRFFDFFDRRBDDBDDBLLDLLDLLDLBBUBBUBB"

    @param cp Corner Permutation
    @param co Corner Orientation
    @param ep Egde Permutation
    @param eo Edge Orientation
    @returns Cube state in the Kociemba facelets representation string

    """
    faces = "URFDLB"
    facelets: List[str] = []
    for i in range(54):
        facelets.append(faces[i // 9])
    for i in range(8):
        for p in range(3):
            facelets[CORNER_FACELET_MAP[i][(p + co[i]) % 3]] = faces[CORNER_FACELET_MAP[cp[i]][p] // 9]
    for i in range(12):
        for p in range(2):
            facelets[EDGE_FACELET_MAP[i][(p + eo[i]) % 2]] = faces[EDGE_FACELET_MAP[ep[i]][p] // 9]
    return ''.join(facelets)
