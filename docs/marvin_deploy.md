# Marvin 真机部署（同步模式）

服务端在推理机上运行 VPP2，客户端在机器人 Orin 上复用 `sharpa-marvin-client` 的观测、QP 预处理、
发布和安全门。两者用该客户端现有的 OpenWAM WebSocket 协议（`ping`、`reset`、`obs_chunk`）通信。

| 文件 | 作用 |
| --- | --- |
| `src/vpp2/marvin_protocol.py` | 协议、观测校验和历史帧，不依赖 torch |
| `src/vpp2/marvin_policy.py` | 把观测转换成训练时的输入，调用模型 |
| `src/vpp2/marvin_server.py`、`scripts/marvin/serve.sh` | 服务入口，`--mock` 不需要权重和 GPU |
| `scripts/marvin/check_deploy.py` | 核对线上输入与训练数据的差异 |
| `scripts/marvin/test_marvin_protocol.py` | 协议和历史帧测试 |
| `deploy/marvin_client/` | 复制到 `sharpa-marvin-client` 根目录的文件 |

## 与训练保持一致的约定

- **动作：** 62 维绝对量，`[左 EEF9, 左手 22, 右 EEF9, 右手 22]`，列 ROT6D，30 Hz。客户端负责行列转换。
- **头部裁剪：** 数据集的头部画面是 `184 120 600 480`，训练时再居中裁到 4:3。客户端默认
  `--video-crop 184 135 600 450` 与之相同。OpenWAM 和 InternW0 的 `184 120 600 480` 会把画面压扁，不要沿用。
- **历史帧：** 条件是片段首帧加 8 帧，间隔 30 步，不足时取首帧。步数按已执行的动作计算，等待推理的时间不计入。
- **每段执行步数：** 只能是 30 的约数（1、2、3、5、6、10、15、30），默认 30，这样每一帧历史都真实观测过。
  32 步动作只执行前 30 步。
- **片段：** 一条指令对应一个片段。换指令必须先 `reset`，首帧也随之更新，客户端需要重新启动。
- **指令：** 支持训练 prompt、数据集里的 `instruction` 和 `Pick up the X with the left/right hand and put it in the bin.`。
  新物体没有缓存的文本特征时，服务端需要加 `--text-encoder`。

## 一、不用权重测试链路

本机或机器人上都可以运行：

```bash
PYTHONPATH=src python -m vpp2.marvin_server --mock --port 8851
```

## 二、在 A800 上启动服务

服务默认只监听 `127.0.0.1`，通过 SSH 隧道访问。`<权重>` 替换为实际的 `step_*.pt`。

```bash
cd /gaozt-test1/lyh/video-prediction-policy-2
export PYTHON_BIN=/gaozt-test1/lyh/envs/vpp2/bin/python
CUDA_VISIBLE_DEVICES=0 scripts/marvin/serve.sh \
  --checkpoint /gaozt-test1/lyh/runs/vpp2_marvin/joint/checkpoints/weights/<权重> \
  --wan /gaozt-test1/lyh/models/VPP2/checkpoints/Wan2.1-I2V-14B-480P \
  --prepared /gaozt-test1/lyh/datasets/vpp2_marvin_pick_place_2_3/prepared
```

```bash
ssh -N -L 8851:127.0.0.1:8851 -p 49427 root@115.191.40.142
```

出现 `VPP2 Marvin server listening` 后，服务才可以使用。

## 三、不接机器人的模拟

`run_vpp2_sim_client.py` 运行真实的策略客户端、同步循环和 QP 预处理，只把 ROS 接口换成数据集回放。
它没有发布器，也不接受 `--execute`。观测按录制数据推进，不受模型动作影响，所以只能检验链路和开环动作误差，
不能代表闭环效果。

需要先把 `deploy/marvin_client/` 复制到 `sharpa-marvin-client`，并让 `PYTHONPATH` 包含 `marvin_gr00t_adapter` 的上级目录。

```bash
python run_vpp2_sim_client.py --media-root <media> --csv <prepared>/val.csv \
  --episode 85 --segment 0 --output <输出目录> --host 127.0.0.1 --port 8851
```

加 `--task` 可以在相同画面上换一条指令。结果写入 `sim_summary.json`，包括延迟、末端位置误差、
首步跳变和被限幅的步数。

## 四、机器人上的 dry-run 和执行

把 `deploy/marvin_client/` 复制到 `/home/nvidia/sharpa-marvin-client` 后运行。不加 `--execute` 就不会发布动作。

```bash
./run_vpp2_client_with_env.sh --host <推理机> --max-loops 5 \
  --task "Pick up the purple coffee capsule with the right hand and put it in the bin."
```

确认 `policy.jsonl` 里每段返回 30 个动作，`command_chunk.jsonl` 里入队 30 个动作，末端目标也合理后，
才可以加 `--execute`。QP 栈、Sharpa 桥和操作步骤与 `sharpa-marvin-client/README.md` 相同。

## 尚未解决

- **RTX 5090：** 权重 32.3 GiB 超过显存 31.8 GiB，服务目前跑不起来，需要先实现分层调度。
- **延迟：** A800 上每次推理约 1.2 秒，长于一段动作的时长。推理期间机器人保持上一个目标，属于走走停停。
- **网络：** 机器人访问 A800 需要能走通隧道或路由，延迟要在现场实测。
