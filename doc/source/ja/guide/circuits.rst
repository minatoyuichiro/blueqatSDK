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

行列から回路へ
--------------

1量子ビットの行列は ``mat1`` でそのまま入ります。2量子ビットは
:func:`~blueqat.decompose.decompose_two_qubit` が分解します:

.. code-block:: python

   from blueqat.decompose import decompose_two_qubit

   c = decompose_two_qubit(matrix)                       # 量子ビット 0 と 1 に
   c = decompose_two_qubit(matrix, targets=(2, 5), n_qubits=6)

大域位相を除いて厳密です。経路は Cartan（KAK）分解

``U = phase * (A1 (x) A2) exp(i(a XX + b YY + c ZZ)) (A3 (x) A4)``

で、相互作用部を ``rxx`` / ``ryy`` / ``rzz`` として出します（一般のユニタリで3つ、
コンパイル後で 6 CX）。**消える正準角は落とす**ので、構造のあるゲートは特別扱い
なしに安くなります。 ``cx`` ・ ``cz`` ・ ``cy`` ・ ``ch`` は回転1つ、 ``iswap``
は2つ、 ``swap`` は3つです。

6 CX は最適の 3 CX ではありません。実機では **CX 予算がおおよそ15個**で結果が
残るかどうかが決まるので、この差は効きます。
:func:`~blueqat.decompose.synthesize_two_qubit` は、解くのではなく
**3CX 回路を勾配降下で目標に当てはめる**ことで 3 CX に届きます:

.. code-block:: python

   from blueqat.decompose import synthesize_two_qubit

   synthesize_two_qubit(matrix)              # ちょうど 3 CX

当てはめは仮定せず\ **検査**\ します（ ``tol`` 以下の不忠実度に届かなければ例外）。
厳密性がゲート数より大事なら閉形式を、実機へ持っていくならこちらを使ってください。

より大きいユニタリと等長写像
----------------------------

:func:`~blueqat.decompose.decompose_unitary` は任意の ``2**n x 2**n`` ユニタリを
Quantum Shannon 分解で、 :func:`~blueqat.decompose.decompose_isometry` は
``2**n x 2**k`` の等長写像を扱います:

.. code-block:: python

   from blueqat.decompose import decompose_unitary, decompose_isometry

   decompose_unitary(matrix)                 # n 量子ビット
   decompose_isometry(v)                     # 入力 k 量子ビット、n まで padding

等長写像の回路は、\ **入力レジスタより上の量子ビットが** ``|0>`` **から始まるとき**\ に
その等長写像を再現します。これは行列積状態を逐次生成回路として書いたときの形そのもので、
各サイトのテンソルがボンドを「ボンド＋新しいサイト」へ写します。

コストは ``4**n`` で増えます。2量子ビットで 3 CX、3量子ビットで 2量子ビットゲート24個、
4量子ビットで120個。\ **最適化された構成ではなく正しい構成**\ なので、実機へ持っていく
回路では先にゲート数を数えてください。

.. note::

   cosine-sine 分解には、scipy が入っていればそれを使います。\ **縮退した場合を厳密に
   扱えるのはそのときだけ**\ です。縮退は例外的な状況ではありません — Toffoli ゲートも、
   等長写像をユニタリへ補完したものも、余弦が重複します。scipy が無い場合の内蔵実装は
   余弦が相異なる行列のみを扱い、それ以外は\ **黙ってユニタリでない因子を返さずエラー**\ にします。

これらより強く最適化された分解が要る場合は、他のツールチェーンの出力を QASM で
取り込んでください:

.. code-block:: python

   # Qiskit 側で、パーサが読める基底へ transpile してから出力する
   #   qc = transpile(circuit, basis_gates=["u", "cx"])
   #   text = qasm2.dumps(qc)

   from blueqat.circuit_funcs.qasm_parser import from_qasm
   c = from_qasm(text)
   c.run(shots=200000, seed=1)

``u`` ・ ``cx`` ・ ``reset`` ・ ``barrier`` ・ ``measure`` はいずれもそのまま
通ります。ただし\ **測定のキーは失われます**\ （OpenQASM 2.0 に置き場所が
ありません）。結果に名前が要る場合は blueqat 側で ``m(key=...)`` を足してください。

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
