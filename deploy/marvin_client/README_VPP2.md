# VPP2 真机测试说明

本目录用于在 Marvin Pro 双臂和 Sharpa Wave 灵巧手上测试 VPP2 的**指令跟随**：同一场景下换一条指令，机器人是否换手、换物体。

- 本目录是 `/home/nvidia/sharpa-marvin-client` 的副本加上 VPP2 文件，原目录没有改动。
- `README.md` 是原客户端的说明，QP、Sharpa 桥等细节可以查它，注意其中的路径是原目录。
- 推理服务在 A800 上运行，Orin 通过 SSH 隧道访问。

> **当前状态：还没有在真机上跑过。** 已验证数据集回放模拟和离线指标，并按第一、二步启动过服务，从 Orin 经隧道 `ping` 成功（往返约 42 ms）。真实相机、ROS 和带图像的推理请求都是第一次测。请严格按顺序来，先 dry-run，再执行动作。

## 运行前确认

- [ ] 急停在手边，有人全程看着机器人，工作区内没有人。
- [ ] **手部序列号已核对。** 下面这条命令显示的序列号必须和实际安装的手一致。Orin 上有两组：`C856903EC855` / `CD549039CD57`（当前配置）和 `C65E9B33C65F` / `CA579B36CA56`（InternW0 配置），目前无法确定装的是哪一组。如果不一致，同时改 `config/marvin_qp.yaml` 和 `config/vpp2_sync.yaml`。

  ```bash
  grep -A2 serials ~/sharpa-marvin-client-vpp2/config/marvin_qp.yaml
  ```

- [ ] 相机和桌面布置与录制数据一致：头部和腕部相机没有移动，bin 放在录制时的位置，物体放在两只手都能安全够到的地方。
- [ ] 没有人在用遥操作。下面第三步会停掉 `apex-teleop.service`。

## 一、启动推理服务

在 Orin 上开**终端 1**，登录 A800。先看哪张卡空闲，服务需要约 38 GiB 显存，把下面的 `CUDA_VISIBLE_DEVICES=0` 换成空闲的卡。

```bash
ssh a800
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
```

```bash
cd /gaozt-test1/lyh/video-prediction-policy-2
export PYTHON_BIN=/gaozt-test1/lyh/envs/vpp2/bin/python
CUDA_VISIBLE_DEVICES=0 scripts/marvin/serve.sh \
  --checkpoint /gaozt-test1/lyh/runs/vpp2_marvin/joint/checkpoints/weights/step_030000.pt \
  --wan /gaozt-test1/lyh/models/VPP2/checkpoints/Wan2.1-I2V-14B-480P \
  --prepared /gaozt-test1/lyh/datasets/vpp2_marvin_pick_place_2_3/prepared \
  --compile
```

启动需要几分钟，等到出现下面这行才算完成。这个终端要一直开着，关掉它服务就停了。

```text
VPP2 Marvin server listening on ws://127.0.0.1:8851
```

## 二、打开隧道

在 Orin 上开**终端 2**。这条命令没有输出是正常的，保持运行即可。

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -L 8851:127.0.0.1:8851 a800
```

客户端默认连 `127.0.0.1:8851`，不需要再传 `--host`。

## 三、启动机器人侧

步骤和原客户端相同，都在本目录运行。先停掉遥操作服务：

```bash
sudo systemctl stop apex-teleop.service
```

**终端 3**，最小 QP 栈：

```bash
cd ~/sharpa-marvin-client-vpp2 && ./run_qp_direct_stack_with_env.sh
```

**终端 4**，Sharpa 桥：

```bash
cd ~/sharpa-marvin-client-vpp2 && ./run_sharpa_bridge_with_env.sh
```

**终端 5**，机器人 Ready 后切到阻抗模式并选择 QP 输入：

```bash
source /etc/apex/apex_ros_env.sh
ros2 service call /tj/control/set_mode marvin_msgs/srv/Int "{data: 3}"
ros2 service call /tj/control/set_input marvin_msgs/srv/Int "{data: 1}"
```

建议从 Home 姿态开始。Home 和录制片段的起始姿态是否接近还没有核对，第一次执行时要留意开头有没有突然移动。

## 四、Dry-run（不发布动作）

不加 `--execute` 就不会发布动作，但客户端会读取真机反馈，并向服务请求动作。`--max-loops` 是请求次数，每次 30 个动作。

```bash
cd ~/sharpa-marvin-client-vpp2
./run_vpp2_client_with_env.sh --max-loops 5 \
  --task "Pick up the blue water bottle with the right hand and put it in the bin."
```

运行结束后查看最新一次的日志：

```bash
LOG=$(ls -dt /home/nvidia/marvin_gr00t_runtime/logs/marvin_* | head -1); echo "$LOG"
python3 - "$LOG" <<'EOF'
import json, math, sys
read = lambda name: [json.loads(line) for line in open(f"{sys.argv[1]}/{name}.jsonl")]
chunks = {c["loop_index"]: c["action_count"] for c in read("command_chunk")}
commands = read("command")
def moved(rows, key):
    a, b = rows[0][key], rows[-1][key]
    return 100 * math.dist([a[i][3] for i in range(3)], [b[i][3] for i in range(3)])
print("段  返回  入队  请求ms  服务ms  左手移动cm  右手移动cm")
for p in read("policy"):
    i = p["loop_index"]
    rows = [c for c in commands if c["loop_index"] == i]
    print(f"{i:2d}  {p['num_steps']:4d}  {chunks.get(i, 0):4d}  {p['inference_latency_sec'] * 1000:6.0f}  "
          f"{p['info']['server_latency_ms']:6.0f}  {moved(rows, 'left_pose_base'):10.1f}  {moved(rows, 'right_pose_base'):10.1f}")
EOF
```

全部满足才能进入下一步：

| 检查项 | 期望 |
| --- | --- |
| 返回、入队的动作数 | 每段都是 30 |
| 服务耗时 | 约 650 ms |
| 请求总耗时 | 记录下来。开发机经隧道约 870 ms，Orin 上还没测过 |
| 末端移动 | 没有异常大的数值。回放模拟里，抓取的手单段移动约 2 到 21 cm，另一只手不超过 2 cm |
| 客户端 | 正常退出，没有报错 |

Dry-run 时机器人不动，画面基本不变，所以这里的末端移动只用来排查明显异常，不代表任务效果。

## 五、真机执行

确认第四步正常后才加 `--execute`。每一步都用手放在急停上，**在终端里按空格**可以停止客户端。第一次先跑很短的几段：

1. **保持不动，3 段。** 观察手臂是否基本不动。这条指令的训练数据只有 8 段，如果手臂明显移动，先停下来记录。

   ```bash
   ./run_vpp2_client_with_env.sh --execute --max-loops 3 --task "Stay still."
   ```

2. **单个物体，3 段。** 看动的手是否正确、方向是否朝向物体。

   ```bash
   ./run_vpp2_client_with_env.sh --execute --max-loops 3 \
     --task "Pick up the blue water bottle with the right hand and put it in the bin."
   ```

3. **完整抓取，15 段。** 训练数据里一次抓放平均约 9 秒动作，也就是约 9 段。客户端不会判断任务是否完成，完成后要按空格。

   ```bash
   ./run_vpp2_client_with_env.sh --execute --max-loops 15 \
     --task "Pick up the blue water bottle with the right hand and put it in the bin."
   ```

**每段之间停顿约 1 秒是正常的。** 模型推理时机器人保持上一个目标，这段时间不属于故障。

**换指令必须重启客户端**，历史帧要从新的首帧开始。推理服务和隧道不用重启。

出现下面任一情况，立即按急停或空格，把日志目录记下来：动作突然跳变、手臂朝人或朝桌面冲、手指持续夹紧不放。

## 六、指令跟随测试

物体优先选训练数据最多的：蓝水瓶、网球、黄色胶带卷、黑色电工胶带卷。每个组合至少跑 3 次，因为每次请求的噪声都不同。每次运行前把机器人和物体恢复到相同的起始状态。

| 组 | 场景 | 指令 | 看什么 |
| --- | --- | --- | --- |
| 基线 | 只放一个物体 | 对应这只手的指令 | 能否抓起并放进 bin |
| 换手 | 同一个物体，同一摆放 | 把 `left` 和 `right` 互换 | 动的手是否跟着变 |
| 换物体 | 两个物体，同一只手都能够到 | 同一只手，换物体名 | 是否去抓指定的物体 |
| 换手又换物体 | 两侧各放一个物体 | 左手抓左边，右手抓右边 | 手和物体是否都对 |

每次运行记录一行：

| 日志目录 | 指令 | 摆放 | 手对吗 | 物体对吗 | 抓起 | 放进 bin | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `marvin_...` | | | | | | | |

**离线结果可以作为预期，但不能代替真机结论。** 这些结果没有物理接触，抓取用距离和手型近似判断：

- 从 A 物体的首帧开始，用模型生成的画面做多步预测：指令为 A 时约 83% 抓起 A，指令换成 B 时约 51% 抓起 B。
- 同一只手、两个目标相距 8 cm 以上时最弱：27 组里，指令 B 有 10 组抓 B，9 组抓成 A，8 组两个都没抓到。
- 片段开头（第一段动作）对指令的响应比抓取过程中弱，所以开头没换手也要多看几段。

## 七、结束和恢复

1. 按空格停止客户端，等它完全退出。
2. 回到 Home。第二条命令会要求输入 `HOME`，完成后 Input Mode 会变成 0。

   ```bash
   ./run_home_reset_with_env.sh --check-only
   ./run_home_reset_with_env.sh
   ```

3. 在终端 3、4 按 `Ctrl+C` 停止 QP 栈和 Sharpa 桥，然后恢复遥操作服务：

   ```bash
   sudo systemctl start apex-teleop.service
   ```

4. 在终端 2 按 `Ctrl+C` 关闭隧道，在终端 1 按 `Ctrl+C` 停止服务并释放显卡。**不用时请关掉服务。**

日志在 `/home/nvidia/marvin_gr00t_runtime/logs/marvin_<时间>/`，每次运行一个目录，测试结束后把记录表和日志目录一起交给模型负责人。

## 可用指令

格式是 `Pick up the <物体> with the left/right hand and put it in the bin.`，文字必须完全一致，另外还有 `Stay still.`。下表括号里是数据集中左手/右手的片段数。

| 两只手都可以 |
| --- |
| blue water bottle (45/37)、tennis ball (34/33)、yellow tape roll (33/34)、black electrical tape roll (28/29)、green sponge block (22/37)、purple coffee capsule (17/16)、blue cup (16/13)、tape measure (14/33)、blackboard eraser (14/12)、chewing gum container (12/6)、black block (11/16)、green formaldehyde-removal gel container (9/7)、red water bottle (7/19)、black dragon figurine (6/5)、tissue pack (5/8)、white spray nozzle (5/8)、white tape roll (5/4)、pink cup (4/9)、red coffee capsule (3/37)、electric screwdriver (3/6)、green cup (3/3)、purple cup (3/3)、glove (1/5)、yellow drink bottle (1/3) |

**只有右手：** yellow cup (4)、red rubber cap (2)、stapler (1)、white spray bottle with green nozzle (1)。

其他物体或手的组合没有缓存的文本特征，服务会拒绝。如果要测，启动服务时加 `--text-encoder`，显存约 49 GiB。这个参数和 `--compile` 还没有一起测过。

## 不要改的参数

`run_vpp2_client_with_env.sh` 已经设好下面这些值，它们必须和训练一致，改错了可能不会报错，模型却会看到错误的输入。

| 参数 | 值 | 原因 |
| --- | --- | --- |
| `--video-crop` | `184 135 600 450` | 和训练的头部画面一致。原客户端文档里的 `184 120 600 480` 会把画面压扁 |
| `--control-hz` | `30` | 训练频率 |
| `--execute-horizon` | `30` | 只能取 1、2、3、5、6、10、15、30，否则历史帧会错位 |
| `--execution-mode` | `sync` | 模型没有按 RTC 或单动作模式训练 |

安全限幅沿用 InternW0 的配置（`config/vpp2_sync.yaml`）：每个动作末端最多移动 10 cm、转动 10°，手部最快 3 rad/s。这个限幅比较宽，**不能代替人工看护**。

## 常见问题

| 现象 | 原因和处理 |
| --- | --- |
| 客户端连不上服务（`Connection refused`、连接超时） | 第一、二步没有完成。确认终端 1 有 `listening`，终端 2 的隧道还在运行 |
| 隧道立刻退出，提示 `Address already in use` | 8851 被占用，多半是旧隧道。用 `ss -ltnp \| grep 8851` 查看 |
| `Expected model_family=vpp2` | 8851 上跑的不是 VPP2 服务 |
| `Unknown task` | 指令格式不对，对照上面的格式 |
| `No cached embedding` | 这个物体和手的组合没有缓存，见“可用指令” |
| `motion readiness failed` | 机器人没有处于阻抗模式或没有选择 QP 输入，重新执行第三步终端 5 的命令 |
| 服务启动时 `CUDA out of memory` | 这张卡被占用了，换一张空闲的卡 |
| 按空格没有反应 | 客户端要在交互终端里前台运行，不要接管道或放到后台 |
