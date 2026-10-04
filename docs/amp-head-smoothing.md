# 独立输出层平滑实验

## 问题与本轮假设

原110/control2500仍是未被替换的基线。上一轮051虽然32/32存活，配对的实际
1ms加速度和力矩增量分别上升4.56%和5.26%；049仅31/32存活，且总体指标更差。
原32固定输入的检查也没有证明时间正则有效。实际时间项梯度远小于主目标，
并且“同输入下的输出更平滑”与“机器人闭环运动更平滑”不是同一证据。

本轮不继续累加旧PPO更新，而从原110的不可变2500状态另建实验，固定CNN、
状态估计器、actor中间层、critic、std与全部旧学习状态，只允许最后
`actor.6.weight[12,128]`和`actor.6.bias[12]`变化。检验冻结特征上的高频输出
是否可被消除而不改变低频步态。若该空间不足，明确拒绝假设，不放松物理验收。

## 数据与求解

- 新reference-only训练seed305、验证seed505；正式各8环境×20秒，短测各4×2秒。
  sealed seed705不能参与拟合。完整66×47输入与128维隐藏量均来自一次真实原策略
  前向，不把CPU重建输入或隐藏量注入模拟器。只有连续、完整、无失败的整episode
  才能用于正式拟合；任一环境失败则拒绝整个队列，不删掉失败样本。
- 原32×60秒只作已知回归测试与排除依据。派生输入
  `resources/amp_inputs/source110_exclusions_v1.npz`为6,477,714字节，
  SHA256 `cf621dc289c74796e37d4d6f547308a17808d4e96c53fb2edb87339f0b326df2`。
  完整保存32个初态、32个首输入和189,920个ready-history的256位指纹与episode
  标签关联。原端点SHA为`de77d4a0c85cd581ae9dc9418d3fbce144730749b9f746f93b9671ec122a833b`，
  两次完整CPU重建与派生文件读回已逐项相等。原始模型、轨迹和视频未改写。
  这只排除精确重复，不证明统计独立；旧history来自CPU重建，不宣称GPU逐位相等。
- 按实际episode边界使用21点Hann有效内区滤波，没有padding、跨episode卷积或
  跨split采样。物理目标为`u=0.5*(W*h+b)`。低频保真与二阶差分平方项使用
  训练原策略的每轴尺度归一化；各episode等权，尺度不接触验证集。
- 预先固定时间项1、source-centered ridge `1e-6`，float64闭式求解一次。
  只根据训练集沿原head到解的单一方向限幅，不对验证集调参或重求解。
  回读实际float32导出参数后重新审核；舍入越界拒绝，不拿float64结果替代。

## 准入不是完成

正式入口所需的真实084流程证书原样保存在
`resources/amp_admission/head_native_smoke_TASK_20261004_084.json`，绑定实际短测
提交`86e71cbf`及全部工件/审计SHA；不复制完整日志或签名URL到Git。
证书保留`formal_admission=false`、`logs_clean=false`与效果/DR未验证状态。
当前入口校验的是新原生流程和实现身份，不要求短测候选数学准入；因此允许
既定8×20秒新采集及唯一求解，但绝不允许采用084拒绝头或跳过新候选的双门禁。
同305/505预声明seed、预算不同的新采集不等于与短测统计独立；现排除集保护
原32和本轮互斥train/validation/sealed，未把084前缀宣称为完全novel。

下述086入口修复改变了driver指纹，故084证书当前仅作历史证据，不再适配修复版。
不得靠改写证书指纹恢复准入；修复版须取得新的真实短测及独立审核证书。

离线门禁：每轴最大目标变化≤0.05rad，低频RMS变化≤0.01rad、峰值≤0.025rad，
有意义轴的低频范围保持98%–102%，触限/溢出计数不增加，全部独立验证episode
的归一化曲率平方至少下降10%。静止轴不能用除零或错误范围比制造改善。

新artifact类型为`actor_head_offline_v1`，完整旧学习状态仅存为不可变source archive；
顶层没有PPO迭代计数或旧证书，不伪装成2750更新或旧训练的续跑。旧31个非head
策略张量及全部Adam/discriminator/replay/未知元数据逐叶字节冻结。

实际执行由Gradmotion注册任务完成，固定1×4090D 24GB。Linux/GPU检查本身不是
平台认证，任务归属、终态、代码与实际产物必须由独立平台/日志审核绑定。
新定义必须先通过真实短测与独立审核，不能复用之前PPO的短测证书。

短测还必须运行新闭环采集入口：仅原110，standing/reference各2环境×1秒，
不能拿候选的离线参数代替这项原生接口验证。训练、验证、head、报告及该实际
录像轨迹分别为`model_7000001.pt`至`model_7000005.pt`；这些编号是上传工件ID，
不是PPO迭代次数。独立短测证书必须绑定第五轨迹的实际SHA和原生日志。

离线通过后仍须原PhysX闭环验证：原32初态×60秒、完整失败前缀、实际1kHz
加速度/力矩变化、足滑、航向、步幅以及5秒、29–30秒、48秒窗口。
还需要未参与拟合的新封存初态对照。不能挑选存活初态或只看低频视频宣称完成。
原DR/noise继续关闭；当前没有训练效果、闭环改善或真机可用结论。

封存协议固定reference seed705、8环境：仅在唯一拟合及float32审核之后采集
原策略20秒的实际输入（`model_7100006.pt`），与原32和新train/validation全部
实际数组重算精确重复排除。该轨迹无论中途是否失败都不进入拟合或修复候选。
随后源策略与候选分别在两个全新原生进程各跑8×60秒，实际12字段初态和3102维
首输入逐条与封存轨迹及彼此匹配，避免同一sim中的接触缓存继承。
配对输出`model_7100007.pt`保留两臂全部8个失败前缀及1ms证据；这是新RSI初态
验证，不宣称跨动作泛化或完整PhysX隐状态逐位相同。

## 当前证据

独立分支`experiment/f1-amp-head-smooth`。首轮完整CPU回归747项通过
（73.766秒，实际exit 0），原证据保留在`outputs/amp-head-smooth/qa-20261004-01/`。
真实短测`TASK_20261004_076`使用发布commit `00ea7ba19ce3b3ad5ff485d72f68f6144c738e96`，
在原生START之前因utils/task_registry/envs循环导入终态6，没有模型产物；不视为
拟合、物理或上传结果。完整日志保存在`outputs/amp-head-smooth/TASK_20261004_076/`。

已修复两个环境初始化入口、角色字段与初始化证据字典的重名，以及原生float32
读回与配置双精度字面值的误比较。角色为`initialization_mode=reference_only`，
`initialization`仍完整保留实际reset/历史清零证据并严格校验；PhysX浮点值要求
精确配置float32表示，dt精确等于原110实际读回，不放宽参数容差或改变物理。
工件目录对齐`logs/<experiment>/exported_data/<run>`，结束后保留60秒扫描时间；
目录和等待本身不是上传证明，仍要求所有实际文件、SHA、回执和平台终态。

正式七文件独立审核已实现：重新核对全部源队列/封存初态、冻结状态与真实1ms
证据，保留全部8个失败前缀及三个指定窗口。缺失窗口明确不可用，不能记为改善。
原32候选检查只是已知回归；封存RSI初态不是跨动作或真机泛化。两臂只导出首输入，
不能宣称闭环每次均值独立重前向；临时臂文件SHA声明不是独立收到的臂文件证明。

修复后的完整CPU回归768项通过（69.636秒，实际exit 0），14个代码/测试文件
在该回归期间保持不变；证据在`outputs/amp-head-smooth/qa-20261004-02/`。
CPU检查不是新的原生成功、正式准入或抖动改善证明。仍需修复版真实短测与正式
实际工件审核；原策略不替换、DR/noise关闭，目标保持进行中。

修复版`TASK_20261004_078`实际使用`b705f5073ca9d050634bd6733871a98a05a91216`，
19:20:32至19:21:00运行28秒，越过循环导入，但在入口复合硬件检查终态6。
原生START/COMPLETE及模型均为0。日志只记录device count 1和torch 2.4.1，
没有GPU名称或CUDA可用性分项，不能据商品标签或旧成功任务补造硬件证明。
完整控制台SHA为`73914900bc530e74bc8e5d6e61b85f9dbe76af1da1868773e334f0ad727d5d3d`。

新增`tools/amp/probe_native_hardware.py`只读回查Linux/CUDA可用性/设备数/实际名称/
显存与compute capability，绑定发布提交及探针自身SHA。它不运行策略、不创建
仿真、不加载模型、不拟合，也不改原硬件门禁；退出0只代表诊断成功，不是短测
准入、抖动改善或DR解锁。

专用`TASK_20261004_080`硬件回读真实终态5、原生诊断进程exit0，运行63秒。
同ESKU000001/4090D商品配置的实际torch名称为`NVIDIA GeForce RTX 4090`（无D），
Linux/CUDA可用/设备数1均正常，显存25393692672B、compute capability 8.9。
这解释了该配置会被旧名称检查拒绝；不鉴定芯片是否D，也不补造078缺失的硬件快照。
完整日志SHA`4b197539cb7d6fed23b62837cdfce9a325bdcc08a70ad81bdb88f08a61d8083a`，
SDK混合收尾状态横幅原样保留。原始平台日志含敏感字段，只保存在本地ignored产物，
分享时仅输出白名单诊断快照；不提交完整日志或签名URL。

运行准入现在严格接受规范化后的两个完整4090/4090D名称，并继续要求Linux、
可用CUDA、真实整数单卡/index0、23–25GiB整数显存与整数compute capability[8,9]。
不选择其他商品、不接受L20/A100/5090/笔记本或多卡；独立平台审核仍只准
实际ESKU000001/gpu1。新`start.native_hardware`与唯一pre-START硬件标记必需、
逐字段精确匹配，缺失或错误类型拒绝，不能把运行名称当作SKU或D硅片认证。

本版791项完整CPU回归通过（84.778秒、实际exit0），证据保存在
`outputs/amp-head-smooth/qa-20261004-03/`；18源码/测试与运行/配置/资产指纹在回归
期间保持不变。新指纹为`5133579691157610a564ce0173a4dc6dad2567431986a72179db808db9328138`。
必须再做第三次真实短测及五工件独立审核，080仅诊断不是可复用短测证书；
第三次短测及后续修复状态如下；不替换原110，不解锁DR/noise，目标未完成。

第三次`TASK_20261004_081`实际使用`ba3beb77492acf3d1faaf441e42ca86d33c93e39`，
19:43:52至19:44:22运行30秒，终态6、模型0。唯一pre-START硬件快照与START
逐字段匹配，原模型字节与实现指纹也匹配；但子采集器拒绝CLI
`--use_gpu_pipeline`（argparse exit2），父进程随后退出，未产生队列、拟合或
录像轨迹。完整日志SHA为`df2d745aa49b04808aabdac59e04fabaea2763cafe6c2416a9498986a079042b`；
该日志仍仅本地保存，不分享含敏感字段的全文。

已将四条原生子入口改为真实解析器支持的`--pipeline gpu`，保留派生属性
`args.use_gpu_pipeline`的运行时门禁。新增CPU AST测试提取四条实际命令构造，
不启动仿真；未改奖励、动力学、源模型或预算。修正仍须新的真实短测与五工件
独立审核，不能将失败081或只读硬件080视为短测通过。

CLI修复版完整CPU回归792项通过（86.856秒、实际exit0），证据在
`outputs/amp-head-smooth/qa-20261004-04/`。18源码/测试在回归期间不变，运行指纹
为`6093f4a672680e959fab3891b9d43a9aa39207895b8aa5508b442b64ebc6c9cc`。
这不是新的原生成功、正式准入或抖动改善证明。

第四次`TASK_20261004_082`使用`a2f0339447718ce08a6cde2ab9c5afafba651e39`，
19:56:28至19:58:11运行103秒，终态6。CLI已被接受，并导出唯一训练队列
`model_7000001.pt`（13,005,152B，SHA
`39ae4eb64e02ae9b2d7d0eca322c428ff6a03af72235364da52222527560e12b`）。
4环境全部完成200步、history与原32去重通过；但父GPU进程的独立重前向在
既定`max(hidden_error,mu_error)<0.001`门禁失败，未进入验证集采集、拟合或录像。
旧异常没有记录两项峰值，不能声称已确定是哪项数值误差。

实际800行的只读CPU检查未复现该失败：batch4与batch256相对记录hidden峰值
约0.000163、mu约0.000160，均无达到0.001的行；全部33个源策略张量不变。
这不是GPU校验通过或短测证书。父重算原为batch256、没有调用source的
`set_seed`；子采集器原为batch4，并用`set_seed`设置cuDNN deterministic与
benchmark。代码可证明上下文未显式统一，不能仅凭CPU结果鉴定GPU失败原因。

现对齐父子原有数值设置及每控制步batch，保存并逐字段核验真实no-grad/eval/
device/cuDNN/TF32/精度/版本读回，失败输出全行峰值位置与两项误差。不修改
TF32或精度设置、不放宽0.001；独立CPU重算继续单列，不冒充GPU或物理证据。
仍需要新定义的真实短测和五工件审核，原模型、奖励、动力学、预算与DR未改。

本次数值上下文修正版完整CPU回归803项通过（82.735秒、实际exit0），
证据在`outputs/amp-head-smooth/qa-20261004-06/`。19源码/测试在回归期间
不变，运行指纹为`eab34c6781c84261bc33e4851c33a2fdee5552da094645e128d64441e738c5d4`。
此前qa-20261004-05的803项有1项失败：原生导入测试桩未提供新调用的
`set_seed`；修正测试桩后重新执行全套回归，不覆盖失败记录。
这仍不是原生短测通过、正式准入或抖动改善证明。

第五次`TASK_20261004_084`使用`86e71cbf71796869ed2f52d7d38f478de705b16a`，
20:21:15至20:23:47运行152秒，实际终态5、SDK exit0。五个独立产物均有
真实SDK上传回执与唯一模型记录，已下载至`outputs/amp-head-smooth/TASK_20261004_084/`。
独立生产审核实际exit0，审计文件SHA为
`6a08625c613974333b27d69da092a5004e0031bc66d1eba0265991f7231be74d`，
事实摘要SHA为`5225a9c354322683d8070090440ef2889e5d302240844a73148cf587c66b5847`。
源train与validation各800行按原batch4重算，GPU hidden/原始mu峰值误差均为0；
独立CPU重算最大误差约0.000201，满足原0.001门禁，仍不冒充GPU数值或物理证据。

此结果只证明原生短测基础流程及证据绑定跑通，**不是平滑候选通过**。
`offline_admitted=false`：float64候选在4个验证初态均未达到曲率改善10%；
float32导出还在训练env2出现输出变化超过0.05rad。没有放宽任何门禁，
没有将该短测候选准入物理评估或PPO续训。日志16条警告被原样保留并令
`logs_clean=false`，不能宣称日志无异常。短测录像仍为原始策略的1秒探针，
不能用于判断5秒、29–30秒和48秒附近的抖动已改善。
正式准入、effectiveness与DR均保持false；原目标仍未完成。

只读补充核验：仅`actor.6.weight`与`actor.6.bias`改变，其余31个策略张量
的shape/dtype/bytes完全一致，旧学习状态归档保留且没有恢复PPO学习。
float32训练env2输出变化峰值为0.05000002058389352rad，严格超过0.05；
验证env0–3曲率平方下降分别5.7781%、5.8562%、5.8119%、5.8420%，均不足10%。
已保存报告的只读诊断显示：未经回退的训练输出变化峰值1.62899rad，所有轴共享
的方向系数为0.0306939；训练env2及全部验证env的最大变化轴为右踝pitch。
共享回退会连带压缩其他关节的修改，这是离线改善偏小的可见机制，不证明实际
1ms关节加速度、触地冲击或闭环抖动的原因。此诊断未重新求解或改变候选。
16条日志警告包含START前的pip/setup/SSL与distutils消息、原生阶段6条
distutils/meshgrid UserWarning，以及SDK Completed之后的wrapper失败文字；
保留实际终态5与exit0，不根据尾部文字反推任务失败，也不声称警告均无害。

### 正式086入口失败与严格历史读取修复

`TASK_20261004_086`实际使用`a4d203477447056cef65f0ca03d7123a597deac6`，
20:47:57至20:48:25运行28秒，平台终态6。原生START、COMPLETE、模型均为0；
实际Git报证书引用的`86e71cbf`提交无法解析，祖先检查exit128。未进入采集或拟合，
故不能将此任务记成数学或物理失败。账单快照runtime=0、deductAmount=0，单位与
最终结算仍未知，不能将平台28秒覆盖账单值。失败任务未重跑或下载不存在的模型。

修复保持原注册driver及原祖先检查：仅在exit128、required commit无法解析、
实际浅仓库、固定GitHub HTTPS origin和当前HEAD/工作区核实后，单次拉取当前HEAD
的64层历史，再验证HEAD/干净状态和原祖先关系。捕获输出避免泄露认证字段，
Git超时、失败或任何不确定状态均拒绝；不切换分支、不跳过证书或放宽门禁。

实际QA07全量824项通过（87.39秒、exit0），含21项新边界测试；运行指纹为
`71073f3f8b26a77cdc5a14fc06078528942c87ffb8b135419f1eda9ac62021dd`。
这是CPU与入口修复证据，不是原生云端或平滑效果证据。084旧证书不适配新指纹，
必须取得新真实短测及独立审核后才能再次启动既定正式队列。奖励、控制、物理、
唯一求解和数学阈值未改，原110/2500与DR/noise关闭状态保留。

### 修复版087真实短测

`TASK_20261004_087`使用修复提交`683fd5bd92b25de8d04780a2f5704a1a98eb25ce`，
21:06:07–21:08:37运行150秒，实际终态5，五份产物及SDK回执齐备。
独立生产审核实际exit0、8.0967秒、stderr=0B，审计文件SHA为
`295f496bd47d1d0c6ca0f866c834ca20e96fb5d8dc52f23ed315499623cf1b74`，
canonical事实SHA为`df1b26f1133fae70db9dac099f1d9db5d73c30b0a9ad10213ab98e54865a24f2`。
五产物、原110、实际硬件、完整日志、代码及新运行指纹均已绑定。

安全流程证书为[`head_native_smoke_TASK_20261004_087.json`](../resources/amp_admission/head_native_smoke_TASK_20261004_087.json)。
这里只保存实际审计输出的31字段，不上传原始日志、SDK令牌、认证字段或签名URL。
`native_verified=true`不代表候选数学准入：`offline_admitted=false`；
`logs_clean/formal_admission/effectiveness_verified/dr_unlocked`均为false。
该候选不继承、不部署，正式队列仍从原110重新采集、只求解一次。
该smoke模式尚不走正式入口的证书历史恢复分支，不能将其成功当作已实测Git恢复。

账单快照唯一087记录扣费0.18、runtime=2、余额18.32；计费单位和最终结算未知，
保留平台150秒与账单runtime的不同含义。终态ledger仅修改087行，非目标字段核验
不变，未切换账号、机器或原模型。原5秒/29–30秒/48秒与全轨迹抖动验收仍未完成。

### 正式02/090仍在原生START前失败

`TASK_20261004_090`实际代码为`4e4352e9b3653a777fa0cd6891cd9efd1a362ac2`，
21:17:22–21:17:57运行35秒、实际终态6。原生START/COMPLETE/history证明及模型均0。
唯一失败观测记录`Head history refresh failed; no retry or diagnostic echo`；结合
实际driver，这对应一次受限历史fetch返回非0。具体stderr未回显，所以认证、
网络、Git选项兼容性仍未知，不能仅凭此推断。硬件marker1不证明策略或采集运行。
完整空模型页、控制台日志及安全诊断保留于`outputs/amp-head-smooth/TASK_20261004_090/`。

没有重复启动90、继承任何失败模型、调整数学或物理门禁。后续先通过受限独立
基础设施诊断获取实际Git版本、选项支持和安全错误分类；不是追加训练/PPO更新，
也不改变原087流程证书或将其当完整效果证明。

独立纯stdlib探针`tools/amp/probe_head_git_history.py`已实现，固定原087证书和
单一注册repo，调用原history helper最多一次，不增加fallback或训练。只输出版本、
帮助中选项是否出现和固定布尔错误分类/返回码/字节长度，不保存或打印transport
原文、URL和认证字段；未知原因仍为unknown。实际新增25项mock测试通过（主线程
0.322秒、exit0），原21项history测试通过（0.081秒、exit0）。纯哈希与导入检查
确认运行指纹仍为`71073f3f`，没有导入Torch/Isaac；不是云端探针结果。

### 091实际Git诊断与兼容修复

独立诊断`TASK_20261004_091`使用`b51fbb7bae0923169a52e9ac895bee95dcf51a63`，
21:38:04–21:38:22实际运行18秒、平台终态5。info/logs单次只读查询exit0；
START、capabilities、transport、COMPLETE各1，严格JSON与代码/证书/运行指纹绑定
均通过。实际Git版本为2.25.1，唯一原helper fetch返回129，stderr固定分类
`unsupported_no_write_fetch_head=true`（2399B，原文不保存或回显），其余分类false。
帮助没列选项本身不算证据；这里有真实fetch拒绝该选项的证据。
认证/网络分类未命中不代表其可用，不能倒填090被抑制的传输原文。

`diagnostic_completed=true`只是诊断结束；helper_succeeded/ancestor_verified/
history_refreshed均false。前后checkout unchanged、单次fetch计数1；没有策略、
采集、拟合、PPO或模型下载，所有离线/正式/效果/DR准入false。控制台2232B SHA
`26086849e0ee6eb80809ecc9ebb782656df6a8b57b9e657ca454785303314d8f`，
安全marker读回留在`outputs/amp-head-smooth/TASK_20261004_091/`。

最小修复仅从driver唯一fetch移除`--no-write-fetch-head`，允许Git正常写
`.git/FETCH_HEAD`元数据，不修改HEAD、分支或工作文件。仍保留exit128+实际缺失
对象+浅仓库条件、固定HTTPS origin、当前HEAD目标、--no-tags/--deepen=64、
前后干净检查、60秒捕获、单次无fallback和最终原祖先核验。奖励、物理、数学
阈值、唯一求解及原110模型未变。修改driver仍会改变运行指纹，新实际纯哈希为
`d16a2f9d28805b41038d00fa5bc58503cb1f6ea71c3a36bb25496ac45bfd325e`。
087证书和b51探针继续保持历史固定；不能改其指纹假装适配新代码。新QA08实际
851项CPU回归通过（93.324秒、exit0），22源码/测试pre/post哈希不变。新增兼容
命令与旧探针拒绝新运行指纹的负例均实际通过；没有本地真实Git fetch/训练/仿真。
完整QA日志SHA为`50f1d2bccab3f1a74412efd0e0cd0dc5e881db9635aad57caad3391dcfcbd382`。
091账单独立快照三类分页完整，唯一记录扣费0/runtime0，gift余额18.32；单位和
最终结算仍未知。root仅更新091终态ledger，其他字段核验不变，原模型与账号不切换。
仍须实际native smoke/独立新证书后才再次运行既定正式队列，当前目标尚未完成。

修复正常发布为`b7f95e2b2522591c48185cf378f0cec8373e399e`；新07短测
`TASK_20261004_092`唯一run请求为2026-10-04T13:50:54.8517437Z。首次只读快照
21:53:11状态3、start21:51:06，账号/单卡SKU/V124与backend完整b7均匹配；
首次快照仅info，尚无原生日志、产物或短测证书证明。该记录不是实时状态声明。
root仅追加092一条运行账本，62/67条；初次读回因checkedAt末位小数零被
PowerShell日期序列化省略而拒绝，随后只读恢复证明wire行完整匹配实际写入模板、
唯一差异为同UTC时刻的格式，其他字段核验不变。没有第二次写入、追加或重启。

092最终实际21:51:06–21:53:36运行150秒、终态5，原生START/COMPLETE/硬件各1，
新b7代码/d16指纹/原110字节与硬件绑定全部核验；五个实际产物和完整日志已下载。
独立生产审核实际exit0、8.0483秒、stderr0；audit83234B SHA
`dd68d1f321007043f2e3b0cbe4497a26937d387111625e94b30558d89544e48f`，
canonical facts SHA`a4963a77f598faea1a430d32d7600ac99e853ea5b8c94943f7c468c7f0d94f43`。
新安全资源[`head_native_smoke_TASK_20261004_092.json`](../resources/amp_admission/head_native_smoke_TASK_20261004_092.json)
2050B/SHA`b099a8adece531c2bb72bdfe16ed473ca95998a86b7f686d5611e3cfa009211e`，
严格JSON、exact31/type/deepvalue与真实审计逐项一致，当前identity/d16与原driver
validator实际通过；本地祖先检查未mock、无fetch，未放宽任何准入。

这仍只证明基础流程。GPU源hidden/rawmu误差0，独立CPU最大约0.000201；冻结31
张量字节精确不变，仅末层2张量修改。候选验证曲率平方下降仍为约5.8%、不足10%；
f32训练env2最大变化0.05000002058389352rad严格超0.05，offline_admitted=false。
16条警告与logs_clean=false保留，源-only1秒recorder不是60秒或指定窗口质量证明。
短测模式仍不执行正式证书history恢复分支，兼容版实际fetch必须在后续正式入口
核验。原110继续保留，不继承短测头、不重跑092；正式队列尚未启动。
092账单独立快照扣费0.18、runtime2、gift余额18.14，单位及最终结算未知；root仅
更新092终态、其他字段核验不变，62/67条。整体抖动目标仍未完成、DR/noise关闭。

### 正式03实际启动请求

2026-10-04 22:10:30.1262447唯一run请求：`TASK_20261004_094`，名称
`f1-amp-head-smooth-v1-formal-20261004-03`。实际create/run均exit0，先保存唯一
request，再完整draft回读与exact dry-run；没有重试、旧任务重跑或账号/机型回退。
固定HEAD`647e797099080c312b6ebce170dc4167b3a75740`/运行指纹d16、092证书b099、
原110/model8802500、V124与一4090D；不续跑PPO、不采用092拒绝头、DR/noise关闭。
预检实际exit0、15.442秒，oldest22:09:02.0602225、expires22:14:02.0602225；
从最老证据计算300秒、绑定commit/指纹/账号池/停用清单，create和run均在窗口内。
原851项实际QA与22当前源码哈希、新证书exact31/实际audit/原模型字节独立核验，
仅为启动准入，不代表新数据或候选效果准入。根唯一ledger append实际exit0、
63/68条、非目标全部字段核验不变；checkedAt使用DateTime避免旧格式误拒绝。
预检/计划/dry-run/request/回读/账本快照保存在ignored
`outputs/amp-head-smooth/launch-formal-20261004-03/`，临时payload已精确清理。
目前只确认平台接收启动请求，实际native/cloudhistory/采集/拟合/门禁/效果待核验；
不得将创建成功、短测结束或对话单轮结束当作抖动修复完成。

22:13:25.3127664首份实际094快照状态3、start22:10:40。backend与原生START均完整
647/d16，原110 SHA与typed GPU绑定通过；正式预算为8×20.0秒、seeds305/505、
PPO0/optimizer_state_reused=false。唯一HISTORY marker实际required b7/head647、
ancestor_verified/history_refreshed=true，首次证明兼容版正式云端历史读取成功。
前置missing ancestor前缀随后已恢复，不当终态错误。训练8段cohort已COMPLETE、
fitEligible/deduplicationPassed=true；验证采集中、stageCOMPLETE0，未声称拟合或
数学通过。首console32774B/SHA1872d704ec48dff79ca1f59f6d7dffe1e80828aa86b74cafeede2d26a5096582。

094第三份info实际终态5、22:10:40–22:15:03运行263秒，backend仍647；完整
Argo505789B/SHA709b04c63d582137acd5208ac2a4852734711a9665522b532fe7256547e60cf4。
全日志严格解析START/COMPLETE/HISTORY/HARDWARE各1，两个cohort各8段合格，
stage记录offline_rejected/solver1/physical/effect/DR false。记录alpha为
0.03084892698216238，各val曲率平方改善约5.787%–5.823%、不足10%，f32 train
峰值0.049999990232437286未超0.05；独立数组/模型审计仍待四实际7100001..4齐全。
预START missing ancestor随后HISTORY已修复；两Traceback均位于stageCOMPLETE后，
白名单类别Pika ConnectionResetError/Errno104。警告保留、不声称logs_clean，
也不将后置传输异常替代数学拒绝原因。原SDK四文件成功receipt计数2/2/1/1。
独立账单快照14:18:38.192255Z三类79/79、0/0、0/0完整，唯一094 goodsUseId
365138617380306944，观察price5.4/runtime4/deductGift0.36/paid0/gift余额17.78；
单位、币种与最终结算未知。根账本当前仍63/68条，下载代理未修改真实池。
本地IWR请求未出现实际timeout异常；前一次0B快照后首件已完成288861152B，
SHA c64db6465ea1469716b03cebf310b70f5889241affc5797c5f10fb67a08bf3b2/CRC通过。
随后仅中断已核实的本地observer，保留第二件125287902B片段，没有停止云任务。
第二件单次curl独占retry01在120.0766秒外部截止，仍收到161939456B进展，
HTTP状态未知；原两个片段保留。下一唯一恢复使用其副本Range续取并严格206/
Content-Range/ZIP CRC核验，不并发重取、重跑云任务或补造缺失模型。

### 正式03完整产物审核与训练瓶颈

094四份实际`model_7100001..4.pt`现已全部完整下载，文件大小依次为
288861152/288859424/93367858/267232B，逐份本地SHA和ZIP CRC通过。
第二件从已保留片段的副本Range续取，实际HTTP206、Content-Range精确匹配；
第三/四件实际HTTP200，三个唯一恢复请求均exit0。原失败片段保留，没有云端
重跑、并发重取、首件重下载或补造模型；平台未提供远端SHA，不声称远端哈希比对。
ignored下载证据位于`outputs/amp-head-smooth/TASK_20261004_094/download-recovery-02/`。

独立生产正式审计实际exit0、17.9369秒、stderr0；审计文件为
`outputs/amp-head-smooth/TASK_20261004_094/native-formal-audit.local.json`，SHA
`c4d561f0283c39e1e43ea8e0df3e37a5ba8e88e1f4bc11f46353248d632b4c56`，
canonical facts SHA `80d4585f1d4f09cbbdee267c0543db546db6a49456aa09f211698265d7a66c2b`。
实际运行绑定仍为完整647提交/d16指纹/原110字节，不改成后续纯文档HEAD。
native/cohort arrays/source forward/head archive/hardware/numerical context均核验通过；
不是synthetic fixture。train/validation各8条完整2000tick、无失败，每组15472个
ready点；整段历史与原32/排除输入/训练组的exact去重均通过，不外推统计独立性。
GPU原源hidden/rawmu误差0，CPU最大差约0.000231、未放宽0.001阈值。
仅actor.6的weight/bias两张量改变，其他31张量dtype/shape/bytes精确不变；
原学习状态只存档、未恢复训练。唯一solver_runs=1，独立auditor_solver_runs=0。

候选仍被原门禁拒绝：8段验证曲率平方改善约5.7873%–5.8226%，均不足10%。
实际导出float32训练峰值0.04999999023243684rad满足0.05界，本轮不能复用092的
舍入超界原因。offline/formal/effect/DR准入均false、未进入已知32或封存新初态
物理评估，不能把文件完整、审计exit0或平台终态5当作抖动修复。
26条日志告警保留、logs_clean=false；其中两次Pika异常位于stage COMPLETE之后，
不抹除，也不将其替代已由真实数组确认的数学拒绝原因。

后续纯训练只读诊断覆盖8×1934×12=185664个ready输出；没有重solve/refit、
修改候选、读取validation选参数或启动额外仿真。记录的float64未缩放训练峰值
1.6208019173215085rad经全局alpha=0.03084892698216238限制到0.05rad，
绑定env2/joint10（right_ankle_pitch_joint）。未缩放float64解及逐点方向未存档，
因此不能认证其极值精确帧；实际存档float32头独立全点复算的唯一极值在
env2/ready-row84/recorded tick150（注册1.5秒），对应输出变化+0.04999999023243684rad。
这是未裁剪策略目标偏移，不是机器人实测关节角或受力证明。
12轴实际训练输出变化峰值依关节顺序约为
0.037112/0.029653/0.011237/0.016704/0.041756/0.010939/
0.044003/0.040701/0.015217/0.014276/0.050000/0.012977rad。
右踝一处峰值使全部12轴统一回退到求解方向约3.0849%，其他轴仍离上限较远。
按实际候选进行训练曲率能量分解，source归一化均值约1、cross=-0.02942198、
step-square=0.00088400，改善约5.796%；说明约5.8%并非采集或入口失败。
这仅解释固定训练输入下的幅度瓶颈，不证明约束改法一定能通过物理验收。
下一独立定义可比较逐轴约束与约束二次求解，仍须预注册、保持原变化/低频/幅度
及曲率门禁、采集新队列并真实短测；不得降低10%门槛或反复利用旧validation调参。

根唯一094终态账本更新实际exit0，status=completed_offline_rejected，63/68条且
所有非目标字段不变；新pool SHA为
`d599e719cfd9ef0d1a4bc2a45f39762e0ac3645637e7981f2d8bdf2587bb36d2`。
观察账单扣gift0.36/runtime4，单位与最终结算未知，不将263秒与账单runtime等同。
原110/control2500不替换，PPO未续跑，DR/noise不解锁，用户指定窗口和全轨迹
抖动目标仍未完成。
