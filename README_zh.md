# Grok ML 实验套件 v3.3（FNAF Track 1）

v3.3：导出出错时不再崩溃，判为 EXPORT_FAILED，可以在同一目录修好后重跑；特征缓存只写了一半时，重跑会自动丢弃半成品重做，已完成的缓存仍然禁止覆盖；测速改用 200 段，并修正了并行分块；特征提取加了 timeout；新增 WAITING_FOR_APPROVAL 和 GIVE_UP 两种消息，重试期间由 Engineer 负责推进，Critic 等待。

## 相比 v3.1 的修复
- **数据完整性检查**：`fnaf/make_task.py` 会核对标签和文件夹 ID 是否一一对应、训练 7000 段 / 测试 8000 段、每段是否都有 frame_0 到 frame_9、每段第一帧能否打开。任何一项不通过就报错退出。
- **特征提取**：已有的缓存不会被覆盖；进程数可以设置（`FEAT_WORKERS`，默认 6）；JPEG 先按缩小尺寸解码，速度更快；先用 50 段测速，并打印全部 15000 段的预计耗时；每次提取的耗时记录在 `cache/<name>.meta.json`。
- **运行器**：概率列里出现非数字时，判为 INVALID 并写明原因，不再直接崩溃；评分时出现任何意外错误也一样处理。RESULT 行新增 precision、recall、被判为异常的比例，以及全判 TRUE 时的 F1（0.667）。删除了 `init --force`，需要修改冻结规则时，在新目录里重新 init。分数口径说明改为"阈值是在同一批折外预测上选的，可能有选择偏差"。
- **Bot 说明**：Engineer 只保留 FNAF 流程，消除了和通用模板的冲突。所有命令统一使用 `/workspace/venv/bin/python`。Critic 增加了失败时如何处理，并且不再提只改阈值的实验。

## 顺序
1. `python3 -m venv /workspace/venv && /workspace/venv/bin/pip install numpy pandas pillow scikit-learn joblib matplotlib`
2. 把本目录放到 `/workspace/comp`，starter kit 放在 `/workspace/fnaf`，数据解压到 `/workspace/fnaf/data/{train,test}`。
3. 给两个 Bot 粘贴 `PROTOCOL.md`、`bots/ENGINEER.md`、`bots/CRITIC.md`。让 Engineer 从 FNAF setup 第 1 步开始做，它会在第 3 步停下等你确认契约。
4. 你确认契约后让它 `init`，并把输出的哈希保存到你的 Mac。然后跑基线，进入循环。15:00 之后运行器会拒绝新实验。

## 提交（由你来做）
只上传 `best/submission.csv`。15:45 前确认最后一次上传的就是它。
