# LOOPS.md：能跑好几天的 Agent 现场笔记

# 让模型来开车：规则简明清单

# Andrej Karpathy，独立研究员

---

**摘要**

这篇文档之所以存在，是因为大多数 agent 系统不是死于模型太弱，而是死于 harness（控制循环）太弱。模型能写代码，能审查代码，能根据十分钟前自己同意的标准来验证自己的输出。但它自己无法决定何时停止、何时重启、结果写到哪里。那是 loop 的工作。本文的模式把 loop 当作一等公民：角色分离，状态落盘，契约在写第一行代码之前就由 agents 协商敲定，出问题时像读 stack trace 一样读 harness。短循环，简单状态，干净契约。其余都是装饰。

> Abstract. This file exists because most agent systems die not from a weak model but from a weak harness. The model can write code; the model can review code; the model can verify its own output against a rubric it agreed to ten minutes ago. What it cannot do, on its own, is decide when to stop, when to restart, and where to write the result. That is the work of the loop. The pattern in this note treats the loop as a first-class object: roles are separated, state lives on disk, contracts are negotiated between agents before the first line of code is written, and the harness is read like a stack trace whenever something goes wrong. Short loops, simple state, clean contracts. Everything else is decoration.

---

**关键词**

agent 循环，Claude Code，harness 设计，生成器-评估器模式，sprint 规划，文件系统状态，契约协商，trace 阅读，可删除脚手架。

> Index Terms. agentic loops, Claude Code, harness design, generator-evaluator pattern, sprint planning, file-system state, contract negotiation, trace reading, deletable scaffolding.

---

## I. 写循环，别写 prompt

prompt 是你敲一次就忘掉的东西。循环是你在睡觉时还在跑的东西。当模型已经好到能在无人监督下执行流程时，杠杆的基本单位就不再是 prompt 了；现在重要的是流程本身。如果你凌晨三点还在反复调一条消息，那你还活在 prompting 时代。关掉那个标签页。写循环。循环很短：收集、推理、行动、验证、重复。本文的全部内容都是这五个动词的注脚。

> A prompt is a thing you type once and forget. A loop is a thing that runs while you sleep. The unit of leverage stopped being the prompt the moment models became good enough to follow a procedure without supervision; what matters now is the procedure. If you find yourself iterating on a single message at three in the morning, you are still in the prompting era. Close the tab. Write the loop. The loop is short: gather, reason, act, verify, repeat. Everything in this document is a footnote on those five verbs.

---

## II. 分离角色

三个角色，三个上下文窗口，三个系统提示。Planner——把一句模糊的人话变成一个 sprint 规格说明，永远不碰代码。Generator——负责写一切，被禁止给自己的作品打分。Evaluator——读 diff，启动 Playwright，操作应用，从第一条消息就被告知"代码是坏的，你的工作是证明它"。混合角色是我见过的最常见失败；模型一旦给自己打分就会变得谄媚，循环悄然收敛于垃圾。

> Three roles, three context windows, three system prompts. A planner that turns a vague human sentence into a sprint spec and never touches code. A generator that writes everything and is forbidden from grading its own work. An evaluator that reads diffs, launches playwright, plays the app, and is told from the first message that the code is broken and its job is to prove it. Mixing the roles is the most common failure I see; the model becomes sycophantic the moment it grades itself, and the loop quietly converges on slop.

---

## III. 先谈契约

Generator 写第一行代码之前，先提出"做完长什么样"，Evaluator 来反驳。双方通过磁盘上的 markdown 文件争论，直到就一份可验证断言的检查清单达成一致。对一个小应用来说，27 条标准是合理的；10 条通常太少，Evaluator 会变成橡皮图章。Planner 给出的原始规格是边界，但契约才是打分的依据。这是让我自己的项目从跑不起来的 demo 变成能用的产品的那一步改变。

> Before the generator writes a single line, it proposes what done looks like and the evaluator pushes back. The two argue via markdown files on disk until they agree on a checklist of testable assertions. Twenty-seven criteria is a reasonable size for a small app; ten is usually too few and the evaluator rubber-stamps. The original spec from the planner is the boundary, but the contract is what gets graded. This is the single change that moved my own runs from broken demos to working products.

---

## IV. 写盘，不写上下文

上下文窗口会说谎。它们会压缩、会腐烂、会把你一小时前说的话藏在一个不是你写的摘要后面。磁盘上的文件不会说谎。维护 `feature_list.json`、`progress.md`、`contract.md`，以及一个只追加的 `log.md`，格式为 `## [YYYY-MM-DD] op | title`。模型应该能在崩溃、丢失会话之后，靠读三个文件就从断点继续。如果你无法用三个文件描述你的状态，那你的状态太复杂了。

> Context windows lie. They compact, they rot, they hide what you said an hour ago behind a summary you did not write. A file on disk does not lie. Keep feature_list.json, progress.md, contract.md, and an append-only log.md with ## [YYYY-MM-DD] op | title entries. The model should be able to crash, lose its session, and pick up where it left off by reading three files. If you cannot describe your state in three files, your state is too complicated.

---

## V. 允许循环重启

看似反直觉——但我从当前前沿模型身上看到的最好行为，是当一次运行跑偏时，它们愿意全扔了重来。老模型修修补补直到代码库像考古地层；新模型，只要有一个干净的 Evaluator 和磁盘上的契约，会在第 9 轮迭代删掉整个项目，然后在第 11 轮交付一个能用的版本。别打断这个过程。重启正是循环正常工作的标志。只有在契约本身出错时才介入人工，而不是编译失败时。

> Counter-intuitively, the best behavior I see from current frontier models is the willingness to throw everything away and start over when a run goes sideways. Older models patched and patched until the codebase resembled archaeology; newer ones, given a clean evaluator and a contract on disk, will delete the project at iteration nine and ship a working version at iteration eleven. Do not interrupt this. The restart is the loop working correctly. Insert a human only when the contract itself is wrong, not when the build is.

---

## VI. 给主观性打分

品味是可以打分的，只要你把它写下来。四个轴，加权：设计、原创性、工艺、功能。用三个 Evaluator 被告知是"好"的参考网站和三个"垃圾"网站来校准。输出是 0 到 1 之间的一个数，以及一段解释差距的文字。模型不会自己发明品味；它只会向你描述的品味收敛。整场游戏在于把评分标准写得足够仔细，使得向它收敛的结果恰好是你真正想要的。

> Taste is gradable if you write it down. Four axes, weighted: design, originality, craft, functionality. Calibrate on three reference sites the evaluator is told are good and three it is told are slop. The output is a number between zero and one and a paragraph explaining the gap. The model will not invent taste; it will only converge toward the taste you described. The whole game is writing the rubric carefully enough that converging toward it is what you actually wanted.

---

## VII. 读 trace

我对 agent 循环的每一个调试洞察都来自阅读原始 transcript，而不是再跑一次实验。把 agent 的输出管道接到一个文件，grep 出它的判断和你的判断分道扬镳的那一刻，为那个精确的时刻修改 prompt，再跑。这和读 stack trace 是同一块肌肉；区别在于 trace 是用英文写的，而且大部分是模型在自言自语。跳过这一步，你就是在凭感觉调参。

> Every debugging insight I have about agent loops came from reading the raw transcript, not from running another experiment. Pipe the agent's output into a file, grep for the moment its judgment diverged from yours, edit the prompt for that exact moment, run again. This is the same muscle as reading a stack trace; the difference is that the trace is written in English and most of it is the model talking to itself. Skip this step and you are tuning by vibe.

---

## VIII. 删掉 harness

harness 是为了补偿模型而存在的。随着模型进步，上季度写的东西有一半变成了开销。会话间的上下文重置对上代模型是承重墙，对下一代是死重；sprint 分解曾经是让四小时构建保持连贯的唯一手段，现在却成了能一次记住两小时的模型的束缚。每次新版本发布，重读你的 harness，删掉模型现在免费就能做到的一切。一个单调增长的 harness，是一个你已经不再阅读的 harness。

> The harness exists to compensate for the model. As the model improves, half of what you wrote last quarter becomes overhead. Context resetting between sessions was load-bearing for one model generation and dead weight for the next; sprint decomposition was the only thing keeping a four-hour build coherent and is now a constraint on a model that holds two hours in one head. Re-read your harness against each new release and delete anything the model now does for free. The harness that grows monotonically is a harness you have stopped reading.

---

## IX. 瓶颈永远在移动

当编码不再是瓶颈，规划就成了瓶颈。当规划被解决，验证就成了瓶颈。当验证自动化，品味就成了瓶颈。你不会完结；你找到下一个要修的东西。loop 的全部意义就是让下一个瓶颈可见。如果一切都很顺利，那是你看得不够仔细。找到新瓶颈，修好它，交付一个更小的 harness，重复。

> When coding stops being the bottleneck, planning becomes the bottleneck. When planning is solved, verification becomes the bottleneck. When verification is automated, taste becomes the bottleneck. You do not finish; you find the next thing to fix. The whole point of the loop is to make the next bottleneck visible. If everything is going smoothly, you are not looking carefully enough. Find the new bottleneck, fix it, ship a smaller harness, repeat.

---

> © 2026 A. Karpathy。允许个人使用本材料。这是将长时运行 agent 循环的工作笔记（loops.md, v060726）独立整理为会议论文格式的版本。自由取用；观点随模型变化而修改。AK

> © 2026 A. Karpathy. Personal use of this material is permitted. This is an independent reformatting of working notes on long-running agent loops (loops.md, v060726) into a conference-style document. Freely available; ideas subject to revision as the models change. AK
