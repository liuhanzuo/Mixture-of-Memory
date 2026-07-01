# HEARTBEAT_LATEST — 上轮状态快照

更新: 2026-07-01 21:00

## ⚠️ HNST v2: 训练停step100, 本机8卡全耗在慢eval
- v2训练主进程(wandb run)停step100(log无error/OOM=非崩, 疑agent设计中途eval检查点)。
- 本机8卡现全跑step100中途eval(needle_recall+fullchain真babilong), 但极慢(44min才11样本, fullchain 0 CSV落盘, 16k流式+oracle重生成拖累)。
- 低效: 训练停+eval占满8卡。已问v2 agent: 主动停还是中断? 要不要eval分.7.53加速腾本机续训。
- v2 step100早期recall(n11极小): 8k v2tree 82%>v1 64%(可训练summary比v1强+18)但<flat 91%<b25 100%。**n11无意义, 待满n。**

## 五线状态
| 线 | 节点 | 状态 |
|---|---|---|
| HNST v2 | 本机 | ⚠️训练停step100, 卡耗在慢eval, 已问agent |
| 训练数据迭代 | .196 8卡 | eval_synth_qa5_readout验证 |
| 解冻ablation | .245 8卡 | 编排器自主wave-1 step145+(5/8layer) |
| 想法3 beacon | B200.55 | 训完跑single-scale eval(qa5 4/8/16k) |
| 解冻验证 | (完成) | 负结论: 解冻不破读出墙45→22 |

## ★★解冻验证终判(已定): 解冻不破读出墙, 根因训练信号→数据迭代线主攻。

## hidden路线三线攻: 数据(训练信号,最押)/v2(选择器树)/beacon(多尺度)。三机制基线: FIFO16k=9/oracle45/reforward52。

## 5节点
本机v2(停+eval) / .196数据 / .245解冻ablation / B200.55 beacon eval / .7.53空(机动, 可接v2 eval分流)

## 下一轮第一件事
1. v2 agent回复→定: 帮它eval分.7.53加速腾本机续训, 还是它有自己逻辑。
2. v2 step100 fullchain端到端(真babilong)聚合 + beacon single-scale结果 + 数据agent验证。
3. 判据出→定最有戏角度。

## 官方判分关键数(全n100除注明, 干净ckpt)
| | raw官方 |
|---|---|
| A模型 hidden FIFO 2k/4k/8k/16k | 54/43/18/9 |
| A模型 fullchain oracle 4k/8k/16k/32k | 35/44/52/58 |
| 解冻sweep fc16k 5层/16层/full | 45/38/22 |
| 选择器16k ra/bm25/oracle | 24/44/43 |
| v2 s100 recall 8k v2tree/v1/flat/b25 | 82/64/91/100(n11!) |
- 锚点pg19 nctx7 16/32k=16/9; ❌b50泄漏作废

## 判分铁律+撤回
官方compare_answers禁re.search; 全档+n100+同设定; 派agent分独占IP+禁泄漏ckpt+防文件冲突; 下结论前查全证据; 合成/小n不算数看真babilong满n; ≤3训练+空节点合16卡。
撤回: W4/mean-pool/slot W2/content-BM25/"slot净负"/re.search端到端/读出墙=多跳/step400=64/方向c修复(平移)/训练16k gap(3584)/H3容量/想法3撞墙/16k容量墙/b50泄漏/HNST树v1零训练KILL(v2继续)/"HNST v2卡死"(误判)/"解冻是主攻"(实测不破读出墙)。
