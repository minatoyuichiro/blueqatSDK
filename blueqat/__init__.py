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
#: So a caller who cares can say so in one line, and fail loudly on a version
#: that would answer backwards, since 2.0.4 has no such attribute::
#:
#:     assert blueqat.BIT_ORDER == "q0_last"
#:
#: `blueqat.backends.backendbase.apply_bit_order` converts, for talking to
#: services that use the other convention.
BIT_ORDER: str = "q0_last"

# 公開するシンボルを明示的に指定（テスト環境の検出をより確実にします）
__all__ = [
    "__version__",
    "BIT_ORDER",
    "Circuit",
    "BlueqatGlobalSetting",
    "Gate",
    "Backend",
    "get_backend",
    "register_backend",
    "TorchBackend",
    "DrawCircuit",
]