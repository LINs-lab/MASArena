# 论文与代码口径核对 / Paper alignment audit

基准：当前 AAMAS 主文与独立附录，2026-10-07。本文记录此次发布准备所修复的实现和材料问题。
作者已确认历史实验结果及无测试参考答案参与推理的条件；旧工作区中发现的路径不用于反推历史实验是否使用过它们。
此包是按当前论文修复的代码与保留实验材料，不是本次重新运行 69,342 个模型任务后的结果。

## 统一口径

| 项目 | 发布口径 |
|---|---|
| 主实验 | 7个配置×10基准×3次运行；每配置每轮3302题 |
| 抽样 | 处理后文件大于400题时 random.Random(42).sample；否则全量；DROP已预先取400 |
| Core | bench_agent；single_agent 是另外的旧实现，不是论文的 Core |
| 模型 | gpt-4.1-2025-04-14；temperature=0.2、top_p=1、8192输出上限；AutoGen critic=0.7 |
| 准确率 | 正确题数/固定评测题数；错误、超时、缺失结果零分；三次均值与样本标准差 |
| pooled | 按各benchmark题数加权，不能替换成十项简单平均 |
| token | 使用实际保留usage；缺失不补零；运行摘要显式报告覆盖率；论文表使用作者保留均值 |
| token取整 | 每benchmark每轮先 ROUND_HALF_UP 到整数，再三次平均和题数加权 |
| GAIA | 单次165题，53/86/26；gpt-4o-mini只用于最终参考答案条件评分 |
| 响应分析 | Jarvis保留165、ChatEval保留163，共328；跨方法可配对163，不能写成330条完整响应 |
| 归因 | 77个Jarvis、74个ChatEval有明确位置的诊断子集，不是完整失败率 |
| 外部CC | 独立运行时案例；工具事件、请求、消息、角色调用、token不是可互换计数 |

## 已修复的实现差异

| 范围 / 文件 | 修复 |
|---|---|
| main.py | manager-tools默认值误装入模型名；模型参数优先级；创建多层结果目录 |
| agents/bench_agent.py | 显式配置不再被构造默认值覆盖；模型上限/采样参数实际传入，避免创建后丢弃 |
| agents/bench_agent.py | 保留明确空工具列表；非检索基准仅Python；HotpotQA/GAIA扩展工具 |
| agents/base.py、workflow_protocol.py | 推理只收到题目、ID、附件；参考答案、测试、评分元数据只留给最终evaluator |
| agents/bench_agent.py | 删除推理期参考答案语义比较与反馈重试；附件与上下文处理保留 |
| agents/bench_agent.py | 每次调用统计增量，共享monitor去重，避免累计值和重复相加 |
| agents/evoagent_newcore.py | 采用已有随附实现的候选答案一致性：非空成功答案strip后精确相等的比例；并列稳定 |
| agents/evoagent_newcore.py | 子代理继承模型/工具/benchmark配置与附件；保留3→6→9、每阶段保留1个和最终5个已有答案，不重跑最终候选 |
| agents/evoagent_newcore.py | crossover/mutation/aggregation调用的已记录usage进入汇总，超时/错误候选不伪造正确分数 |
| agents/jarvis_newcore.py | 按附录恢复planning→selection→dependency-aware execution→response四阶段；未知模型/无效依赖显式报错 |
| agents/llm_debate_newcore.py | 同构两个角色按轮交替，3轮后汇总；继承同一benchmark与工具配置 |
| agents/autogen_newcore.py | 最多5轮Primary/Critic，不因APPROVE文本提前停止；返回最后Primary答案 |
| agents/camel_newcore.py | 保留3轮与TASK_FINISHED退出规则；模型、工具和usage参数统一 |
| agents/chateval_newcore.py | 保留Math/Logic/Critical两轮顺序；提示词限制与真正工具权限区分，保留最终提取usage |
| benchmark_runner.py | 固定分母；错误标记不能保留旧正确分数；初始化失败变成零分结果而非中断整批 |
| benchmark_runner.py、agents/base.py | 缺失usage不记零，失败时保留已收集usage；dict消息不再用不可哈希对象生成ID |
| benchmark_runner.py | pass@k正确读取AgentResult；不改变论文pass@1设置；局部抽样RNG不污染工作流随机状态 |
| evaluators/drop_evaluator.py、hotpotqa_evaluator.py | F1阈值0.3转换成二值成功；低于阈值不给部分准确率分数 |
| evaluators/math_evaluator.py、aime_evaluator.py | 明确final_answer优先；空答案不算正确；数学数值容差为绝对1e-3，无默认相对容差 |
| evaluators/bbh_evaluator.py、utils/answer_extraction.py | 小写/括号选项一致；HotpotQA专有名词不误当选择题或布尔答案 |
| evaluators/bbh_evaluator.py | word_sorting逐词按顺序比较，并保留重复词次数；删除set比较，乱序或缺少/多出重复词均判错；仅忽略大小写与空白格式差异 |
| evaluators/ifeval_evaluator.py | 严格评估收到未strip、未截断的完整答案；空/不完整指令列表不能全通过 |
| evaluators/base_code_evaluator.py | MBPP测试环境保留test_imports；仅加载指定评测文件，不强依赖未分发的训练/公开测试文件 |
| evaluators/gaia_evaluator.py | 字符串false不再按Python非空字符串转为True |
| evaluators/base_evaluator.py、math_evaluator.py | 尊重自定义数据路径；MATH移除隐藏重采样；无须恢复本机旧目录才能执行 |
| evaluators/utils/normalization.py | GAIA level保留到输出，供后续按层分析 |
| uv.lock、requirements.txt | 与pyproject当前项目名一致，已有版本锁不升级；离线锁检查通过 |
| README.md、configs/paper、scripts/prepare_paper_data.py、run_paper.py | 替换旧等权单次结果；固定文件校验值、抽样索引和210条运行命令 |

## 附录过程统计：经用户确认后按现存日志重算

唯一历史过程缓存差异来自ChatEval L1批次20260209_195838；L2/L3一致。
已同步附录表S11/S13及analysis/gaia_trace_stats下的CSV，主实验正确数和token表未改变。
所有每题均值仍按作者确认的165评测分母展示，保留响应数单列163。

| 指标 | 原附录 | 更新后 |
|---|---:|---:|
| Step external | 4623 (28.02/题) | 5953 (36.08/题) |
| Executed external | 3916 (23.73/题) | 4925 (29.85/题) |
| Direct/code markers | 1189 (7.21/题) | 1476 (8.95/题) |
| Retrieval-step | 1836 | 2338 |
| Executed retrieval | 3544 | 4449 |
| Executed file tools | 168 (1.02/题) | 231 (1.40/题) |
| Structured code-tool | 2787 | 3615 |

旧log.rar的5方法统计、GSM8K分析、163题配对、CC逐文件统计、归因步骤分布均可由保留输入重算；不把不同日志粒度强行合并。

## 打包与匿名

完整公开包和匿名完整包包含所有定位到的当前附录支持材料；25MB内的投稿包保留代码、原始表、728份响应、结构化CC事件、归因和重算脚本，完整CC请求dump及历史材料在完整包。
所有版本均清除凭据形态文本、本机用户路径和工作簿作者元数据。匿名版另去掉pyproject作者字段；论文参考文献与第三方版权声明保留。
RAR中的日志以清洗后的普通文本分发；ZIP/TAR内部记录也递归清洗。原始输入不修改。
每包含MATERIALS.json、MANIFEST.sha256和VALIDATION.json；当前paper/scripts/build_paper.sh不会再用仅PDF的ZIP覆盖完整材料包。

## 验证边界

BBH顺序修复增加3个回归测试，覆盖乱序、重复词缺失/多出以及正确序列的格式归一化；先复现旧实现误判，再验证修复。作者已确认历史BBH评测保留顺序，本次修复发布代码，不重评分或修改历史结果。

离线回归实际执行运行/评分/调度逻辑，以假客户端隔离外部模型工具；这不等于安装499个依赖后的真实API端到端测试。
未重跑付费模型实验；修复会影响后续运行结果，不将旧统计伪称新代码测量值。
Qwen检查能证明整数题数与汇总自洽，但不替代缺失原始运行记录；GLM的L2/L3有作者确认正确数，未独立恢复其他未保留来源。
完整benchmark语料、GAIA附件、API账户和外部CC运行时须按各自来源获取；固定数据hash只描述提供的本地处理文件，未重新下载验证上游字节一致性。
跨论文所有可见表和所列材料已核对；任何未保留的原始记录、真实provider账单或未运行的环境行为不能据此声称已验证。

## 2026-10-09 publication review

- AutoGen propagates Primary failures and marks Critic exceptions with a top-level
  error while retaining observed messages and usage, so incomplete workflows receive
  zero credit under the existing runner scoring rule.
- `main.py --max-completion-tokens` provides an explicit positive output cap.
  `scripts/run_paper.py` pins all 210 commands to the protocol's 8,192-token cap;
  explicit settings take priority even when `MAX_TOKEN_SIZE` is invalid.
- Six offline regressions cover these failure and configuration paths; 85 paper
  tests pass. No paid model run or historical-result recomputation was performed.
- This branch excludes dataset records, logs, trajectories and generated memory.
  Dataset download helpers and reproducibility manifests remain available.
