MCPサーバ（LLM連携）
====================

blueqat は `MCP (Model Context Protocol) <https://modelcontextprotocol.io/>`_
サーバを同梱しており、Claude Desktop や Claude Code などのLLMクライアント
から自然言語で量子回路の構築・実行・解析・描画ができます。

セットアップ
------------

.. code-block:: console

   pip install blueqat[mcp]

その後、 ``blueqat-mcp`` コマンドをMCPクライアントに登録します。
Claude Desktop なら設定ファイルに:

.. code-block:: json

   { "mcpServers": { "blueqat": { "command": "blueqat-mcp" } } }

Claude Code なら:

.. code-block:: console

   claude mcp add blueqat -- blueqat-mcp

ツール一覧
----------

``run_circuit(qasm, shots=None, backend="tensornet")``
   OpenQASM 2.0 回路を実行。小さい回路は状態ベクトル、幅の広い回路は
   上位の基底状態確率、 ``shots`` 指定時は測定カウントを返します。

``circuit_stats(qasm)``
   量子ビット数・深さ・ゲート数。

``expectation_value(qasm, hamiltonian)``
   :math:`\langle\psi|H|\psi\rangle` 。ハミルトニアンは
   ``"1.5*Z[0]*Z[1] - 0.5*X[0] + 2"`` のようなパウリ式で指定します。

``draw_circuit(qasm)``
   回路図をPNG画像で返します。

``eo_transpile(qasm)``
   論理回路をExchange-Onlyスピン量子ビットのパルスへコンパイルし
   （:doc:`exchange_only` 参照）、パルススケジュールを要約します。

``blueqat_info()``
   バージョンと機能の一覧。

安全性
------

ツール入力がコードとして実行されることはありません。回路は eval を使わない
OpenQASMパーサで、ハミルトニアンは :func:`blueqat.utils.parse_hamiltonian`
（正規表現ベースの小さなパーサ）で解釈されます。状態ベクトルの応答には
上限があり（幅の広い回路は確率の要約を返す）、1回の呼び出しでクライアント
が溢れることはありません。
