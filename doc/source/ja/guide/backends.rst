バックエンドと実行
==================

シミュレーションモード
----------------------

1つのシミュレータに2つの実行モードがあります:

- ``tensornet`` （デフォルト）: ``opt_einsum`` によるテンソルネットワーク
  縮約。要求されない限り全状態ベクトルを実体化しないため、幅が広く浅い
  回路では密シミュレーションよりはるかにスケールします。
- ``statevector``: 密な状態ベクトルの時間発展。

.. code-block:: python

   Circuit(20).h[:].run()                      # tensornet (デフォルト)
   Circuit(20).h[:].run(backend='statevector') # 密
   Circuit(20).h[:].run(mode='statevector')    # 同上

両モードは数値的に一致し、どちらもautogradのグラフを保持します。

返り値
------

.. code-block:: python

   c = Circuit(2).h[0].cx[0, 1]

   c.run()                                   # 状態ベクトル (torch.Tensor)
   c.statevector()                           # 同上 (明示)
   c.m[:].run(shots=100)                     # ビット列のCounter
   c.shots(100)                              # 同上 (明示)
   c.run(amplitude='11')                     # 単一の確率振幅
   c.m[:].oneshot()                          # (収縮後の状態, 1つの測定結果)
   c.expect(hamiltonian)                     # <psi|H|psi>
   c.probs([1])                              # 周辺確率

大規模回路
----------

密な状態は ``2**n`` 要素あります。 ``tensornet`` モードで28量子ビットを
超える回路は、全ベクトルの代わりに ``shots=`` か ``returns='amplitude'``
を指定してください:

.. code-block:: python

   Circuit(50).h[:].run(shots=3)
   Circuit(50).h[:].run(returns='amplitude', amplitude='0' * 50)

サンプリングは逆CDF探索を使っており、カテゴリ数の上限はありません。

再現可能なサンプリング
----------------------

``seed=`` は実行中のすべての乱数（ショットサンプリング、回路途中の
collapse、大規模 ``n`` のperfect sampling）を固定します。同じ回路・同じ
シードなら常に同じcountsになります:

.. code-block:: python

   c = Circuit(4).h[:]
   c.run(shots=200, seed=42) == c.run(shots=200, seed=42)   # True

シードは ``torch.manual_seed`` ではなく専用の ``torch.Generator`` を駆動する
ため、回路にシードを与えてもプログラムの他の部分が使う乱数には影響しません。
``seed=`` を渡さなければ、従来通りランダムなままです。

countsからの推定
----------------

**件数の少ないもの同士の比は推定になりません。**\ 3万試行で失敗が0件のとき、
失敗率は0ではなく「おおよそ 1e-4 より下」です。それを割り算に使うと、0が分子と
分母のどちらに来るかで、0になったり意味のない跳ね上がりになったりします。
割る前に\ **最低件数を決めて明示**\ してください:

.. code-block:: python

   failures = counts.get('1', 0)
   if failures < 10:
       rate = None          # 比を取るには事象が足りない
   else:
       rate = failures / shots

2つの実行を比べるときも同じです。どちらの率も数件の事象から作られているなら、
総ショット数がいくら多くてもその比はほとんど情報を持ちません。

countsのビット順
----------------

既定ではcountsのキーの左端が最大番号の量子ビットで、 ``key[-1]`` が量子
ビット0です。クラウドAPI（ ``qapi.blueqat.app`` を含む）は逆の並びを使う
ため、 ``bit_order='q0_first'`` を指定すると各自で反転させる代わりにその
並びで受け取れます:

.. code-block:: python

   Circuit(3).x[0].run(shots=4)                          # Counter({'001': 4})
   Circuit(3).x[0].run(shots=4, bit_order='q0_first')    # Counter({'100': 4})

キーはどちらの並びでも必ず ``n_qubits`` 桁にゼロ埋めされます。ゼロ埋めが
ないと、反転した ``'11'`` が量子ビット0,1なのか4,5なのか区別できないから
です。他所で得た ``Counter`` には
:func:`~blueqat.backends.backendbase.apply_bit_order` で同じ変換をかけられます。

中間測定とリセット
------------------

``reset`` とキー付き測定は「いつ収縮が起きるか」に結果が依存するため、
そのような回路は自動的にショットごとの量子軌道シミュレーションとして
実行され、各 ``measure`` / ``reset`` でその場で収縮します:

.. code-block:: python

   Circuit(2).h[0].cx[0, 1].reset[0].m[:].run(shots=100)

   Circuit().x[0].m(key='a')[0].run(shots=3, returns='samples')
   # [{'a': [1]}, {'a': [1]}, {'a': [1]}]

カスタム初期状態
----------------

.. code-block:: python

   import torch
   psi0 = torch.tensor([0, 1, 0, 0], dtype=torch.complex128)
   Circuit(2).h[0].run(initial=psi0)

その他の組み込みバックエンド
----------------------------

- ``'draw'`` -- matplotlibによる回路図。
- ``'draw_tn'`` -- 回路のテンソルネットワークグラフ。
- ``'eo'`` -- Exchange-Onlyトランスパイラ (:doc:`exchange_only` 参照)。
- ``'cloud'`` -- クラウド送信 (:doc:`cloud` 参照)。
- ``'1q_compaction'`` / ``'2q_decomposition'`` -- 1量子ビットゲートの統合 /
  2量子ビットゲートの基底変換を行うトランスパイラ。

独自バックエンドの登録
----------------------

.. code-block:: python

   from blueqat import register_backend, Backend

   class MyBackend(Backend):
       def run(self, gates, n_qubits, *args, **kwargs):
           ...

   register_backend('mybackend', MyBackend)
   Circuit(2).h[0].run(backend='mybackend')
   Circuit(2).h[0].run_with_mybackend()      # 同等
