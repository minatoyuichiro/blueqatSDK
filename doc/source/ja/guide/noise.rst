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
で状態が完全混合状態に置き換わる）。2量子ビットゲートの後では、そのゲートの
**両方の量子ビットに一括で**（4種のパウリ積の混合として）作用します。
もう一つの流儀は ``p`` を「何らかのパウリ誤りが起きる確率」と読むもので、
そちらは1量子ビットチャネルの :func:`~blueqat.noise.pauli_depolarizing` です。
両者は ``p_pauli = 3 * p / 4`` のとき一致します。**取り違えると全ての数値が
ずれる**ため、引数で切り替えるのではなく別の名前にしてあります。

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

=========  ==================  ==============
量子ビット  密度行列            1ゲートあたり
=========  ==================  ==============
8          1 MB                0.4 ms
10         17 MB               9 ms
12         268 MB              183 ms
=========  ==================  ==============

10量子ビット程度までは快適、12まで実用的です。14を超えるとメモリを使い切る
前にバックエンドが拒否します。
