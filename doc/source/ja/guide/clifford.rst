Clifford演算子
==============

:class:`~blueqat.clifford.Clifford` は演算子を行列ではなくスタビライザ
タブロー（ ``X_0..X_{n-1}`` と ``Z_0..Z_{n-1}`` の共役による像）として保持
します。そのため合成と逆元がビット演算だけで厳密に計算でき、 ``2**n`` の
配列はどこにも現れません。

.. code-block:: python

   from blueqat import Circuit
   from blueqat.clifford import Clifford, random_clifford

   c = Clifford.from_circuit(Circuit(2).h[0].cx[0, 1])
   c.to_circuit()          # ゲートへ戻す: h, s, sdg, cx, x, z
   c.inverse()
   c.then(other)           # c を先に、そのあと other

Cliffordゲート集合は ``i`` ・ ``x`` ・ ``y`` ・ ``z`` ・ ``h`` ・ ``s`` ・
``sdg`` ・ ``sx`` ・ ``sxdg`` ・ ``cx`` ・ ``cy`` ・ ``cz`` ・ ``swap`` です。
それ以外（ ``t`` や ``rx`` など）は**黙って近似せずエラー**にします。
大域位相は追跡しません（観測できず、ベンチマークで使うCliffford群は位相を
除いて定義されるためです）。

一様ランダムなClifford
----------------------

.. code-block:: python

   random_clifford(2, seed=0)     # 2量子ビットの11520個から一様に

一様性は、シンプレクティック基底を共役ペアごとに作ることで得られます。
``X_i`` の像はまだ使える非恒等パウリから一様に、 ``Z_i`` の像はそれと反交換
するものから一様に、残りは両方と交換するものから選びます。この選択肢の数を
掛け合わせるとちょうど ``|Sp(2n, 2)|`` になり、残りのパウリ因子は ``2n`` 個の
独立な符号ビットが与えます。 ``seed`` は専用の generator を使うので、
グローバルな乱数には影響しません。

ランダム化ベンチマーキング
--------------------------

タブローが要る理由がこれです。ベンチマークの系列は、**それ以前を全部打ち消す
たった1つのCliffford**で終わらなければなりません。系列を逆順に流し直すと
長さが約2倍になり、別のものを測ることになります。

.. code-block:: python

   from blueqat import Circuit
   from blueqat.clifford import Clifford, random_clifford
   from blueqat.noise import depolarizing

   def rb_circuit(n, m, seed):
       total = Clifford.identity(n)
       circuit = Circuit(n)
       for i in range(m):
           c = random_clifford(n, seed=seed * 1000 + i)
           circuit += c.to_circuit()
           total = total.then(c)
       return circuit + total.inverse().to_circuit()

   # 生存確率: 雑音がなければ厳密に1、雑音があると減衰する
   rho = rb_circuit(1, 16, seed=0).run(noise=depolarizing(0.02))
   survival = float(rho[0, 0].real)

``survival`` を系列長 ``m`` に対してフィッティングする部分はベンチマークの
解析そのものなので、SDKではなく実験を行う側に置いてあります。
