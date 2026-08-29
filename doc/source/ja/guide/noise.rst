雑音と密度行列
==============

:meth:`~blueqat.circuit.Circuit.run` に ``noise=`` を渡すと、回路は密度行列
シミュレーションに切り替わり、各ゲートの直後にチャネルが適用されます:

.. code-block:: python

   from blueqat import Circuit
   from blueqat.noise import depolarizing

   rho = Circuit(2).h[0].cx[0, 1].run(noise=depolarizing(0.01))

``noise=`` を渡さなくても ``run(backend='density')`` で密度行列バックエンドを
使えます（純粋状態の ``|psi><psi|`` が返ります）。

チャネル
--------

.. code-block:: python

   from blueqat.noise import (depolarizing, pauli_depolarizing,
                              amplitude_damping, phase_damping, kraus)

   depolarizing(p)          # (1-p) rho + p I / 2**k
   pauli_depolarizing(p)    # (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
   amplitude_damping(gamma) # |1> が |0> へ減衰    (T1)
   phase_damping(lam)       # エネルギーを失わずコヒーレンスだけ失う (T2)
   kraus([k0, k1, ...])     # Kraus演算子を直接渡す任意のチャネル

:func:`~blueqat.noise.depolarizing` は Nielsen & Chuang の定義です（確率 ``p``
で状態が完全混合状態に置き換わる）。もう一つの流儀は ``p`` を「何らかのパウリ
誤りが起きる確率」と読むもので、そちらは
:func:`~blueqat.noise.pauli_depolarizing` です。両者は ``p_pauli = 3 * p / 4``
のとき一致します。\ **取り違えると全ての数値がずれる**\ ため、引数で切り替えるので
はなく別の名前にしてあります。

2量子ビットゲートの後、減極は既定ではそのゲートの\ **両方の量子ビットに一括で**
（``4**k`` 個のパウリ積の混合として）作用します。 ``per_qubit=True`` を指定
すると、代わりに1量子ビットのチャネルを各量子ビットへ独立に当てます:

.. code-block:: python

   depolarizing(0.02)                   # cx の後に一括の2量子ビットチャネル
   depolarizing(0.02, per_qubit=True)   # 各量子ビットへ1量子ビットチャネル

**両者は物理的に別の写像で、どちらにも用途があります。**\ 一括の方は k量子ビット
チャネルを普通に書いたもので、\ **純粋に局所的な雑音を仮定する論文**\ が意味して
いるのは独立の方です。同じ強度なら独立の方がやや強く減衰します（1つで済んで
いたところに2つのチャネルが当たるためです）。

減衰チャネルは1量子ビットのチャネルで、ゲートが触れた各量子ビットに適用され
ます。測定・リセット・バリアには雑音を載せません。

雑音モデル
----------

チャネルを1つ渡すと全ゲートの後に適用されます。ゲートごとに誤り率を変えたい
場合（実機では2量子ビットゲートの誤りが一桁大きいのが普通です）は、ゲート名を
指定します:

.. code-block:: python

   from blueqat.noise import NoiseModel, depolarizing, amplitude_damping

   nm = NoiseModel()
   nm.add(depolarizing(0.001))                     # 全ゲートの後
   nm.add(depolarizing(0.01), gates=['cx', 'cz'])  # さらにこれらの後
   nm.add(amplitude_damping(0.002))

   Circuit(3).h[0].cx[0, 1].run(noise=nm)

``noise=`` はチャネルのリストも受け取ります（順に適用されます）。

準静的雑音
----------

シリコンスピン量子ビットの主要な位相緩和は、マルコフ的なチャネルではありません。
核スピン（Overhauser）場や 1/f 電荷雑音は回路の実行より\ **はるかにゆっくり**\ 揺らぐ
ため、1回の繰り返しの中では detuning がほぼ一定で、繰り返しにわたる平均が
デコヒーレンスになります。この\ **時間相関は Kraus 演算子では表現できません**\ 。
そして違いは学術的なものではなく、実測できます:

.. code-block:: python

   from blueqat.noise import QuasiStatic

   Circuit(1).h[0].i[0].i[0].run(quasi_static=QuasiStatic(sigma=0.4),
                                 samples=4000, seed=1)

各サンプルは量子ビットごとに1つの detuning を ``N(0, sigma)`` から引いて固定し、
回路の各層の後にすべての量子ビットへ ``rz(delta_q * dt)`` を積みます。得られた
密度行列を平均するので、これは detuning についての古典的な混合そのものです。
自由誘導減衰はガウス型 ``exp(-(sigma * t)**2 / 2)`` になります（ ``t`` は層数）。

決定的な検証は\ **ハーンエコー**\ です。待ち時間の中央での反転は、その前に溜まった
静的なずれを打ち消しますが、記憶を持たないチャネルには何の効果もありません。
同じ回路・同じ総待ち時間での実測コヒーレンス:

.. list-table::
   :header-rows: 1

   * - 雑音
     - エコー無し
     - エコー有り
   * - ``QuasiStatic(0.4)``
     - 0.02
     - **0.92**
   * - ``phase_damping(0.25)``
     - 0.32
     - 0.27

つまり T2* やエコーの実験を :func:`~blueqat.noise.phase_damping` で再現しようと
すると、\ **強度をどう調整しても答えは合いません**\ 。

``samples`` は平均する detuning の本数です（既定200）。誤差は ``1/sqrt(samples)``
で減ります。準静的雑音はチャネルと併用できます（ ``quasi_static=`` と ``noise=``
の両方を渡してください）。

雑音の強度を変える
------------------

``noise_scale=c`` は全チャネルの強度を ``c`` 倍します。ゼロ雑音外挿（ZNE）が
回すのがこのつまみです。同じ回路を複数の雑音強度で実行し、期待値をゼロへ
外挿します:

.. code-block:: python

   import numpy as np
   from blueqat.utils import Z

   c = Circuit(2).h[0].cx[0, 1]
   h = 1.0 * Z[0] * Z[1]
   scales = [1.0, 2.0, 3.0]
   values = [float(c.run(noise=depolarizing(0.02), noise_scale=s, hamiltonian=h))
             for s in scales]
   zero_noise = np.polyfit(scales, values, 1)[-1]   # 強度0へ外挿

強度パラメータが有効範囲を外れるスケールを指定した場合は、黙って丸めずに
エラーになります（丸めた点は外挿を静かに壊すためです）。

``noise_scale`` は準静的雑音にも効きますが、 ``sigma`` を ``c`` 倍ではなく
``sqrt(c)`` 倍します。ガウス型の位相緩和は ``exp(-(sigma t)**2 / 2)`` で減衰する
ので、外挿が線形になるのは ``sigma**2`` の側だからです。

雑音つき実行の結果
------------------

.. code-block:: python

   c.run(noise=nm)                      # 密度行列（2**n x 2**n のテンソル）
   c.run(noise=nm, shots=1000, seed=1)  # 対角成分からサンプリングしたcounts
   c.run(noise=nm, hamiltonian=h)       # Tr(rho H)

``shots`` は状態ベクトルのバックエンドと同じ ``seed`` ・ ``bit_order`` を
受け取ります。 ``returns='statevector'`` ・ ``'amplitude'`` ・ ``'samples'``
は状態ベクトルの概念なので、ここでは拒否されます。

計算量
------

密度行列は ``4**n`` 個の要素を持ち、どのゲートも全要素に触れるため、この
バックエンドは小さい回路向けです。ゲートとその直後のチャネルは1枚の演算子に
掛け合わせて1回で適用しており（Kraus演算子を個別に当てるより約8倍速い）、
それでもスケーリングは変わりません:

.. list-table::
   :header-rows: 1

   * - 量子ビット
     - 密度行列
     - 1ゲートあたり
   * - 8
     - 1 MB
     - 0.4 ms
   * - 10
     - 17 MB
     - 9 ms
   * - 12
     - 268 MB
     - 183 ms

10量子ビット程度までは快適、12まで実用的です。14を超えるとメモリを使い切る
前にバックエンドが拒否します。
