已新增独立的“两台机器、每台 2 张昇腾 A3”配置，原来的 16 卡文件没有修改。

文件：

- [4 卡服务端配置](/Users/linjiajia/project/twinkle/cookbook/client/server/transformer/server_config_dsv4_0731_npu_2node_2npu.yaml)
- [4 卡服务端启动脚本](/Users/linjiajia/project/twinkle/cookbook/client/server/transformer/run_dsv4_0731_npu_2node_2npu.sh)
- [4 卡客户端脚本](/Users/linjiajia/project/twinkle/cookbook/client/twinkle/run_dsv4_0731_npu_2node_2npu_client.sh)

主节点 `172.61.10.111`：

```bash
DSV4_MODEL_ID=/你的/减层模型路径 \
NETWORK_IFACE=eth0 \
RESET_RAY=1 \
bash cookbook/client/server/transformer/run_dsv4_0731_npu_2node_2npu.sh head
```

Worker 节点 `172.61.12.165`：

```bash
DSV4_MODEL_ID=/你的/减层模型路径 \
NETWORK_IFACE=eth0 \
RESET_RAY=1 \
bash cookbook/client/server/transformer/run_dsv4_0731_npu_2node_2npu.sh worker
```

客户端：

```bash
DSV4_MODEL_ID=/你的/减层模型路径 \
DATASET_ID=/model/ljl/dataset/self-cognition.jsonl \
OUTPUT_DIR=/shared/twinkle_output/dsv4-0731-a3-2node-4npu \
bash cookbook/client/twinkle/run_dsv4_0731_npu_2node_2npu_client.sh
```

两个 IP 已写成默认值，通常不需要再传 `HEAD_IP` 或 `NODE_IP`。配置为：

```text
每节点 NPU：2
节点数：2
总 rank：4
FSDP size：4
EP size：4
默认 batch size：4
```

`DSV4_MODEL_ID` 已直接接入 YAML 环境变量解析，因此替换模型时不需要修改 YAML，只需确保主节点、Worker 和客户端传入相同值。语法和配置拓扑校验均已通过。