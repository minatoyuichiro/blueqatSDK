# Copyright 2019-2026 The Blueqat Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Blueqat Quantum Computing SDK core module."""

# _version.py からバージョン情報を引っ張ってくる
from blueqat._version import __version__

# 1. コアクラスとグローバル設定を公開
# (BlueqatGlobalSetting を circuit からインポートして追加します)
from blueqat.circuit import Circuit, BlueqatGlobalSetting
from blueqat.gate import Gate

# 2. バックエンド関連の絶対インポート
from blueqat.backends.backendbase import Backend, get_backend, register_backend
from blueqat.backends.torch_backend import TorchBackend
from blueqat.backends.draw_backend import DrawCircuit

#: Which end of a counts key is qubit 0. blueqat writes ``"q0_last"``: the
#: *rightmost* character is qubit 0, so ``Circuit(3).x[0].m[:].run(shots=1)``
#: gives ``{'001': 1}``.
#:
#: It is exported because it has not always been this. blueqat 2.0.4's numpy
#: and numba backends put qubit 0 at the *left*, giving ``{'100': 1}`` for the
#: same circuit -- the mirror image, with no error and no warning. Code that
#: reads a bitstring is therefore version-dependent in a way nothing announces,
#: and both versions call themselves blueqat.
#:
#: ⚠ Its absence does not identify a version. It was added after 2.1.3 was
#: released, so an installed 2.1.3 -- which answers ``q0_last``, correctly --
#: has no such attribute either. Treating "no attribute" as "2.0.4" would
#: reject a working install.
#:
#: The check that holds on every version measures instead of asking::
#:
#:     order = getattr(blueqat, "BIT_ORDER", None) or blueqat.measure_bit_order()
#:     assert order == "q0_last"
#:
#: `measure_bit_order` runs one three-qubit circuit and reads the answer, so it
#: is right about whatever is actually installed, including versions that
#: predate both it and this constant -- copy its two lines rather than
#: importing it, if the code has to run against those.
#:
#: `blueqat.backends.backendbase.apply_bit_order` converts, for talking to
#: services that use the other convention.
BIT_ORDER: str = "q0_last"


def measure_bit_order() -> str:
    """Which end of a counts key is qubit 0, established by running a circuit.

    Returns ``"q0_last"`` or ``"q0_first"``. Asking costs one shot of a
    three-qubit circuit and is the only answer that holds across versions:
    `BIT_ORDER` did not always exist, and a version that lacks it may still be
    a correct ``q0_last`` install.

    The equivalent two lines, for code that must also run on a version without
    this function::

        c = Circuit(3); c.x[0]
        order = "q0_last" if "001" in c.m[:].run(shots=1) else "q0_first"
    """
    circuit = Circuit(3)
    circuit.x[0]
    key = next(iter(circuit.m[:].run(shots=1)))
    if key == "001":
        return "q0_last"
    if key == "100":
        return "q0_first"
    raise RuntimeError(
        f"the probe circuit answered {key!r}, which is neither '001' nor "
        f"'100'. Something other than the bit order is wrong.")

# 公開するシンボルを明示的に指定（テスト環境の検出をより確実にします）
__all__ = [
    "__version__",
    "BIT_ORDER",
    "measure_bit_order",
    "Circuit",
    "BlueqatGlobalSetting",
    "Gate",
    "Backend",
    "get_backend",
    "register_backend",
    "TorchBackend",
    "DrawCircuit",
]