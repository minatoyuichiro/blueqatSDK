微分可能な回路・VQE・QAOA
=========================

シミュレータを通した勾配
------------------------

任意のゲートパラメータに ``requires_grad=True`` の :class:`torch.Tensor`
を渡せます。ゲート行列・状態の時間発展（両実行モード）・確率・期待値まで、
パイプライン全体が微分可能なtorch演算で構成されています:

.. code-block:: python

   import torch
   from blueqat import Circuit
   from blueqat.utils import Z

   theta = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
   energy = Circuit(1).rx(theta)[0].expect(1.0 * Z[0])
   energy.backward()
   theta.grad        # -sin(0.4)、厳密な解析勾配

このため変分アルゴリズムにパラメータシフト則は不要で、通常の
``torch.optim`` のオプティマイザがそのまま使えます。

パウリ演算子とハミルトニアン
----------------------------

:mod:`blueqat.utils` がパウリ代数を提供します:

.. code-block:: python

   from blueqat.utils import X, Y, Z, I, from_qubo, qubo_bit

   h = 0.5 * Z[0] * Z[1] + 1.2 * X[0] - 3.0
   h = h.simplify()
   h.to_matrix(2)                   # 密/疎のtorch行列
   term = (X[0] * Y[1]).to_term()
   evo = term.get_time_evolution()  # exp(-i t P) を回路に追加する関数

``from_qubo`` はQUBOのコスト行列をIsingハミルトニアンに変換します。

VQE
---

.. code-block:: python

   import torch
   from blueqat import Circuit
   from blueqat.utils import AnsatzBase, Vqe, Z, X

   class MyAnsatz(AnsatzBase):
       def get_circuit(self, params):
           return Circuit(2).rx(params[0])[0].ry(params[1])[1].cx[0, 1]

   hamiltonian = (1.0 * Z[0] * Z[1] + 0.5 * X[0]).simplify()
   ansatz = MyAnsatz(hamiltonian, n_params=2)
   result = Vqe(ansatz).run()
   result.most_common(4)

``Vqe`` は任意の ``torch.optim`` オプティマイザクラス、オプションの
サンプラ（ショットベース推定の ``get_measurement_sampler(n)`` 、厳密で
勾配を保つ ``non_sampling_sampler`` ）、 ``initial_params`` を受け取れます。

実行の再現と収束の確認
----------------------

``initial_params`` を渡さない場合、 ``Vqe.run()`` はランダムなパラメータから
始まるため、同じ問題でも実行ごとに異なる局所最適解に落ちます。QAOAでは
「最適解が見つかる確率」が実行ごとに大きく変わることもあります。 ``seed=``
を渡すと実行全体が決定的になります:

.. code-block:: python

   Vqe(ansatz, seed=42).run()          # または: Vqe(ansatz).run(seed=42)

1つのシードが両方の乱数を決めます。初期パラメータと、
``get_measurement_sampler(n, seed=...)`` で作ったシード対応サンプラの
サンプリングです。 ``Circuit.run(seed=...)`` と同じく専用の generator を
使うので、グローバルな乱数には影響しません。

各実行は反復ごとの目的関数値を記録するので、別のオプティマイザで回し直さ
なくても収束の様子を確認できます:

.. code-block:: python

   result = Vqe(ansatz, seed=42).run()
   len(result.loss_history)      # 実際に回った反復数
   result.loss_history[-1]       # 最後に記録された目的関数値

ショット雑音とパラメータシフト則
--------------------------------

ショットから期待値を推定すると autograd のグラフが失われます（カウントは
数値であって、ゲート角の微分可能な関数ではありません）。そのため
ショットベースの目的関数には逆伝播できる勾配がありません。 ``Vqe`` は
それを検知して**パラメータシフト則**に切り替えます。同じ推定器を、
ずらしたパラメータで評価することで勾配を得る方法です:

.. code-block:: python

   from blueqat.utils import get_measurement_sampler

   vqe = Vqe(ansatz, sampler=get_measurement_sampler(2000, seed=3), seed=42)
   result = vqe.run()          # 動く。逆伝播だけでは動かない

各ゲートの寄与は ``(E(theta + pi/2) - E(theta - pi/2)) / 2`` で、有限差分の
近似ではなく**厳密**です。これを autograd でアンザッツのパラメータへ連鎖
させるので、QAOAの角のように**1つのパラメータが多数のゲートを動かす**場合も
寄与が正しく合算されます。

``gradient=`` で選択を上書きできます。 ``'backprop'`` は常に逆伝播
（ショットサンプラでは失敗します）、 ``'parameter_shift'`` は常にシフト則、
既定の ``'auto'`` は目的関数が微分可能かどうかで選びます。シフト則は
パラメータ付きゲート1適用あたり2回の追加実行を要するので、厳密経路での
逆伝播の代わりに無条件で使うものではありません。

厳密に成り立つのは、生成子の固有値が2つでその差が1のゲートに限られます:
``rx`` ・ ``ry`` ・ ``rz`` ・ ``p`` / ``phase`` ・ ``rxx`` ・ ``ryy`` ・
``rzz`` ・ ``cp`` ・ ``exch`` 。制御回転（ ``crx`` ・ ``cry`` ・ ``crz`` ）は
固有値が4つで4項の規則が必要なため、**黙って誤った勾配を返さずエラー**に
します。

:func:`~blueqat.utils.parameter_shift_gradient` は同じ仕組みを直接公開して
おり、任意のアンザッツとエネルギー推定器に対してエネルギーと勾配を返します。

QAOA
----

:class:`~blueqat.utils.QaoaAnsatz` は、項が互いに可換なハミルトニアン
（自動チェックされます）から標準的なQAOAアンザッツを構築します:

.. code-block:: python

   from blueqat.utils import QaoaAnsatz, Vqe, from_qubo

   qubo = [[1, 1], [1, 0]]
   h = from_qubo(qubo)
   ansatz = QaoaAnsatz(h.simplify(), step=2)
   result = Vqe(ansatz).run()
   print(result.most_common(2))

完全な自己検証付きプログラムはリポジトリの ``examples/maxcut_qaoa.py`` と
``examples/vqe_ground_state.py`` を参照してください。
