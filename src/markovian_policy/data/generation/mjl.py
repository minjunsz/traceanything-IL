"""Parser for MuJoCo .mjl logs (haptix record format), as written by the human teleoperation setup."""

import struct
from dataclasses import dataclass
from typing import Final

import numpy as np

from markovian_policy.arrays import FloatArray

HEADER_BYTES: Final = 28  # nq, nv, nu, nmocap, nsensordata, nuserdata, name_len (int32 each)


@dataclass(frozen=True)
class MjlLog:
    qpos: FloatArray  # (T, nq)
    qvel: FloatArray  # (T, nv)
    ctrl: FloatArray  # (T, nu) position-actuator targets


def parse_mjl(data: bytes, skip: int = 1) -> MjlLog:
    """Every `skip`-th record of a log. A record is [time, qpos, qvel, ctrl, mocap pos/quat, sensordata, userdata]
    in float32. Raises ValueError for a truncated log."""
    nq, nv, nu, nmocap, nsensordata, nuserdata, name_len = struct.unpack("7i", data[:HEADER_BYTES])
    body = data[HEADER_BYTES + name_len :]
    record = 1 + nq + nv + nu + 7 * nmocap + nsensordata + nuserdata
    if len(body) % (4 * record):
        raise ValueError("truncated log")
    records = np.frombuffer(body, dtype=np.float32).reshape(-1, record)[::skip].astype(np.float64)
    return MjlLog(records[:, 1 : 1 + nq], records[:, 1 + nq : 1 + nq + nv], records[:, 1 + nq + nv : 1 + nq + nv + nu])
