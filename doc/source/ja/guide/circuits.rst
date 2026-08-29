回路とゲート
============

回路の構築
----------

:class:`~blueqat.circuit.Circuit` は操作のリストを保持します。ゲートは属性、
量子ビットは ``[...]`` で選択し、パラメータ付きゲートはインデックスの前に
呼び出しでパラメータを渡します。すべてチェーンできます:

.. code-block:: python

   import math
   from blueqat import Circuit

   Circuit().h[0].cx[0, 1].rz(math.pi / 4)[1].m[:]

量子ビット0は常に状態ベクトルインデックスの最下位ビットです（ ``'10'`` は
量子ビット1が1、量子ビット0が0。QiskitのStatevectorと同じ規約です）。

ゲートセット
------------

1量子ビットゲート
   ``i``, ``x``, ``y``, ``z``, ``h``, ``s``, ``sdg``, ``t``, ``tdg``, ``sx``,
   ``sxdg``, ``phase(theta)`` （別名 ``p``, ``r`` ）, ``rx(theta)``,
   ``ry(theta)``, ``rz(theta)``, ``u(theta, phi, lam[, gamma])``,
   ``mat1(matrix)`` （任意の2x2ユニタリ）。

2量子ビットゲート
   ``cx`` （別名 ``cnot`` ）, ``cy``, ``cz``, ``ch``, ``swap``, ``iswap``,
   ``iswapdg``, ``cphase(theta)`` （別名 ``cp``, ``cr`` ）, ``crx``, ``cry``,
   ``crz``, ``cu(theta, phi, lam[, gamma])``, ``rxx(theta)``, ``ryy(theta)``,
   ``rzz(theta)``, ``zz``, ``zzdg``, ``exch(theta)`` （ハイゼンベルク交換
   パルス。:doc:`exchange_only` を参照）。

3量子ビットゲート
   ``ccx`` （別名 ``toffoli`` ）, ``ccz``, ``cswap`` 。

その他の操作
   ``m`` / ``measure`` （ ``m(key="name")`` でキー付き中間測定）, ``reset``,
   ``barrier`` 。

パラメータを取らないゲートにパラメータを渡すと ``ValueError`` になります
（例: ``x(0.5)[0]`` は黙って無視されず拒否されます）。

回路の情報取得
--------------

.. code-block:: python

   c = Circuit(3).h[:].cx[0, 1].cx[1, 2].m[:]
   c.n_qubits      # 3
   c.depth()       # 4  (並列ゲートは1段と数える。barrierは数えない)
   c.count_ops()   # Counter({'h': 3, 'cx': 2, 'measure': 3})

測定確率（微分可能・指定量子ビットへの周辺化に対応）とハミルトニアン
期待値:

.. code-block:: python

   from blueqat.utils import Z

   Circuit(2).h[0].cx[0, 1].probs()          # tensor([0.5, 0., 0., 0.5])
   Circuit(2).h[0].cx[0, 1].probs([1])       # 量子ビット1の周辺確率
   Circuit(1).rx(0.4)[0].expect(1.0 * Z[0])  # <Z> = cos(0.4)

観測量には任意のパウリ式が使えます（和も可）:
``c.expect(1.0 * Z[0] * Z[1] - 0.5 * X[2])`` 、
``c.run(hamiltonian=...)`` でも同じです。値はハミルトニアンを
``2**n x 2**n`` の行列にせず、状態ベクトル上で項ごとに計算するため
``O(項数 * 2**n)`` で済み、行列形式が現実的でなくなる13量子ビット付近を
大きく超えて使えます。

パウリ指数
----------

:meth:`~blueqat.circuit.Circuit.exp_pauli` は、パウリ積 ``P`` に対する
``exp(-i * theta * P)`` を追加します。Trotter分解や化学・QAOAのアンザッツの
基本部品です。演算子は「量子ビット番号 → パウリ文字」の辞書で与えるので、
ビット順の曖昧さがなく、疎な積も短く書けます:

.. code-block:: python

   Circuit().exp_pauli({0: 'X', 1: 'X', 2: 'Z', 3: 'Y'}, 0.3)  # exp(-0.3i XXZY)
   Circuit().exp_pauli({5: 'Z'}, t)                            # == rz(2t)[5]

``P**2 == I`` なので、これは厳密に ``cos(theta) - i sin(theta) P`` です。
規約（1/2 を付けない）は :meth:`~blueqat.utils.Term.get_time_evolution` と
同じで、そちらは ``Term`` から同じ列を組み立てます。 ``theta`` には
``torch.Tensor`` を渡せる（パラメータが微分可能なまま）ほか、 ``'I'`` の
項は無視されます。

逆回路
------

:meth:`~blueqat.circuit.Circuit.dagger` はエルミート共役（ゲートを逆順に
して共役化）を返します。測定とリセットには逆操作がないため例外になります
が、 ``dagger(ignore_measurement=True)`` なら除去して続行します:

.. code-block:: python

   c = Circuit(3)  # ... 構築 ...
   identity = c + c.dagger()   # |0...0> に逆計算で戻る

OpenQASM 2.0
------------

.. code-block:: python

   qasm = Circuit(2).h[0].cx[0, 1].to_qasm()

   from blueqat.circuit_funcs import from_qasm
   c = from_qasm(qasm)

JSONシリアライズ
----------------

回路はバージョン付きのJSON互換スキーマでラウンドトリップできます
（クラウド送信のワイヤ形式でもあります）:

.. code-block:: python

   from blueqat.circuit_funcs.json_serializer import serialize, deserialize

   data = serialize(Circuit(2).h[0].cx[0, 1])
   c = deserialize(data)

回路描画
--------

``run(backend='draw')`` でmatplotlibによる回路図を描画します。登録済みの
全ゲートが描画可能で、未知の（ユーザー登録）ゲートは ``UserWarning`` 付き
で省略されます。

名前付きゲートブロック
----------------------

実際のアルゴリズムはサブルーチンの入れ子です — Shorの位数発見は
「初期化・制御剰余乗算・逆QFT」で構成され、それぞれがさらに小さな部品から
できています。名前付きブロックはその構造を回路オブジェクトに保持したまま、
実行には一切影響しません（各バックエンドは内部のゲートを透過的に見ます）:

.. code-block:: python

   c = Circuit(7)
   with c.block("order-finding"):
       with c.block("superposition"):
           c.h[4, 5, 6]
       with c.block("c-U^1"):
           c.cswap[4, 2, 3].cswap[4, 1, 2].cswap[4, 0, 1]
       # ライブラリ回路をブロックとして配置 (量子ビット4..6へシフト)
       c.append_block("IQFT", qft_circuit(3).dagger(), offset=4)

   print(c.tree())
   # Circuit(7)
   # └─ order-finding
   #    ├─ superposition
   #    │  └─ h[4, 5, 6]
   #    ├─ c-U^1
   #    │  └─ ...
   #    └─ IQFT
   #       └─ ...

ブロックは任意に入れ子でき、 ``repr()`` と :meth:`~blueqat.circuit.Circuit.tree`
に現れ、 :meth:`~blueqat.circuit.Circuit.dagger` では鏡像ブロック
（ ``"order-finding†"`` ）として保存されます。 ``depth()`` / ``count_ops()``
は中身のゲートを数え、 ``flatten()`` / JSONシリアライズはブロックを
プレーンなゲート列に展開します（フラットなワイヤ形式には階層は残りません）。
回路描画ではブロックは関与する量子ビットにまたがる1つの名前付きの箱として
描かれます。回路全体が単一のブロックに包まれている場合は自動でその中に
降り、子ブロックが箱として表示されます。 ``expand_blocks=n`` で n 階層
ぶん展開、 ``expand_blocks=True`` で全ゲートまで展開できます。
完全なプログラム例は ``examples/shor_15.py`` を参照してください。

アンシラ量子ビット
------------------

.. code-block:: python

   c = Circuit(4).h[:]
   with c.ancilla() as a:        # 新しい量子ビットを確保
       c.cx[0, a[0]]
       c.cx[0, a[0]]
   # ブロックを出るとアンシラは |0> にリセットされます (reset=True がデフォルト)

マクロとカスタムゲート
----------------------

関数を回路メソッドとして、あるいはゲートクラスをゲートセットに登録できます:

.. code-block:: python

   from blueqat import BlueqatGlobalSetting
   from blueqat.decorators import circuitmacro

   @circuitmacro
   def bell(c, a, b):
       return c.h[a].cx[a, b]

   Circuit(2).bell(0, 1)

   BlueqatGlobalSetting.register_gate('mygate', MyGateClass)
