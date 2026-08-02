クラウドアクセス
================

:mod:`blueqat.cloud` は、Blueqatクラウドサービス（``https://qapi.blueqat.app``）
へ回路を送信します — マネージドなシミュレータと実機量子コンピュータを、
ローカル実行と同じ ``Circuit`` API で使えます。

APIキー
-------

キーは https://mcp.blueqat.app/login で取得できます。認証情報は次の
優先順位で解決されます:

1. 現在のプロセスでの ``blueqat.cloud.configure(api_key=...)``
2. 環境変数 ``BLUEQAT_API_KEY``
3. 設定ファイル ``~/.blueqat/config.json``

.. code-block:: python

   import blueqat.cloud as cloud

   cloud.save_api_key("YOUR_API_KEY")   # 所有者のみ (0600) の権限で保存
   cloud.me()                           # アカウントのプラン・上限・残クォータ

クラウドでの回路実行
--------------------

:mod:`blueqat.cloud` をインポートすると ``'cloud'`` バックエンドが登録
されます。結果はSDKのローカル規約に揃えられるため、そのまま置き換えられます:

.. code-block:: python

   import blueqat.cloud
   from blueqat import Circuit
   from blueqat.utils import Z

   c = Circuit(2).h[0].cx[0, 1]
   c.m[:].run(backend='cloud', shots=100)          # Counter('00', '11', ...)
   c.run(backend='cloud')                          # 状態ベクトル (torch.Tensor)
   c.run(backend='cloud', amplitude='11')          # 単一の確率振幅
   c.run(backend='cloud', hamiltonian=1.0 * Z[0])  # 期待値

名前付きブロックとスライスはワイヤ形式向けに自動展開され、ハミルトニアンの
恒等（定数）項はローカルで加算し直されます。

その他のエンドポイント
----------------------

.. code-block:: python

   cloud.health()                  # サービス稼働確認 (キー不要)
   cloud.circuit_info(c)           # サーバ側での検証・統計
   cloud.vqe_run(h, n_qubits=2)    # クラウドでVQE
   cloud.qaoa_run(qubo_terms)      # QUBOのQAOA

実機量子コンピュータ
--------------------

.. code-block:: python

   cloud.hardware_status()         # 実機ステータス (公開・キー不要)
   cloud.hardware_qpus()           # 利用可能なQPU一覧 (要認証)
   cloud.submit_hardware_job(c, shots=100, confirm=True)

``submit_hardware_job`` は ``confirm=True`` が必須です: 実機実行は実費が
発生し、アカウントのクォータの対象になります。

MCP連携
-------

同梱の :doc:`MCPサーバ <mcp>` は ``cloud_run_circuit`` と
``cloud_hardware_status`` ツールを公開しており、APIキーを設定した
LLMクライアントからローカルだけでなくクラウドでも回路を実行できます。

テスト
------

HTTPトランスポートは注入可能で、ネットワークなしのテストが書けます::

   cloud.configure(transport=lambda method, path, payload, key, endpoint: {...})
