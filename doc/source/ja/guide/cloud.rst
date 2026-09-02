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

   cloud.hardware_status()         # QPU の状態（鍵なしで見られます）
   cloud.hardware_qpus()           # 使える QPU の一覧
   cloud.hardware_calibration()    # 量子ビットごとの誤り率とコヒーレンス時間
   cloud.hardware_next_window()    # 次に投入を受け付ける時間帯
   cloud.hardware_quote(shots=100, payer="me")   # 投入する前に費用を見る

   job = cloud.submit_hardware_job(c, shots=100, confirm=True)

   cloud.hardware_jobs()                        # 最近投入したジョブ
   cloud.hardware_job(job["task_id"])           # 1件の状態
   cloud.hardware_job_result(job["task_id"])    # 終わっていれば結果
   cloud.cancel_hardware_job(job["task_id"])    # 待ち行列にいるうちなら取消

``submit_hardware_job``\ は\ ``confirm=True``\ を必ず求めます。実機は実際に費用が
かかり、アカウントの割り当ての範囲でしか動かないためです。書いたとおりの量子ビット
番号を保ちたいときは\ ``preserve_layout=True``\ を渡してください（既定では
サービス側が割り当て直します）。

動いたかどうか分からないとき
----------------------------

通信の失敗が、いつも「実行されなかった」を意味するとは限りません。このサービスは
Cloudflare の後ろにあり、524 は\ **応答が始まらないまま 100 秒ほど沈黙した**\ ときに
出ます。これは「処理にかけてよい時間の上限」ではなく「沈黙の上限」です。要求自体は
既に届いていて、しかも終わっている可能性があります。手元の待ち時間切れも同じです。

この場合は素のエラーではなく :class:`~blueqat.cloud.CloudOutcomeUnknown`\ が
上がるので、「起きなかった」と「起きたかどうか分からない」を区別できます。

.. code-block:: python

   try:
       job = cloud.submit_hardware_job(c, shots=100, confirm=True)
   except cloud.CloudOutcomeUnknown:
       # そのまま投げ直さないでください。既に待ち行列にいるかもしれず、
       # 実機ジョブの二重投入は枠と費用をもう一度使います。
       for j in cloud.hardware_jobs()["jobs"]:
           print(j["task_id"], j["status"])

``CloudOutcomeUnknown``\ は\ ``RuntimeError``\ の子なので、既に
``RuntimeError``\ を捕まえている書き方はそのまま動きます。接続を拒否された場合は
従来どおり普通のエラーです。何も送られていないので、動いているはずがないからです。


MCP連携
-------

同梱の :doc:`MCPサーバ <mcp>` は ``cloud_run_circuit`` と
``cloud_hardware_status`` ツールを公開しており、APIキーを設定した
LLMクライアントからローカルだけでなくクラウドでも回路を実行できます。

テスト
------

HTTPトランスポートは注入可能で、ネットワークなしのテストが書けます::

   cloud.configure(transport=lambda method, path, payload, key, endpoint: {...})
