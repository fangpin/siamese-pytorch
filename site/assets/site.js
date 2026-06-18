const DEFAULT_LANGUAGE = "en";
const STORAGE_KEY = "siamese-homepage-language";

const translations = {
  en: {
    "brand.eyebrow": "Siamese · Omniglot · PyTorch",
    "brand.title": "Siamese Networks for One-Shot Learning",
    "brand.subtitle": "PyTorch reimplementation on the Omniglot dataset",
    "nav.docs": "Docs",
    "hero.eyebrow": "One-shot learning system",
    "hero.title": "Twin encoders, one-shot decisions.",
    "hero.body":
      "A compact Siamese PyTorch implementation trained on Omniglot, with a clear path from paper to runnable code.",
    "hero.ctaRepo": "GitHub Repository",
    "hero.ctaDocs": "Documentation",
    "hero.ctaQuickStart": "Quick Start",
    "metrics.label": "Quick facts",
    "metrics.accuracy": "Final accuracy",
    "metrics.evaluation": "One-shot test",
    "metrics.framework": "Framework",
    "metrics.dataset": "Dataset",
    "architecture.label": "Architecture snapshot",
    "architecture.note":
      "Shared twin encoders, absolute difference, then a final linear score.",
    "quickStart.eyebrow": "Run the project",
    "quickStart.title": "Quick Start",
    "quickStart.body":
      "Prepare Omniglot, create the model directory, and launch training with the provided script.",
    "notes.eyebrow": "Implementation notes",
    "notes.title": "Why the result differs from the paper",
    "notes.optimizerTitle": "Optimizer choice",
    "notes.optimizerBody":
      "This implementation uses Adam instead of SGD with momentum.",
    "notes.paramsTitle": "Parameter settings",
    "notes.paramsBody":
      "The code keeps default PyTorch initialization and shared settings instead of layer-specific tuning from the paper.",
    "curve.eyebrow": "Experiment artifact",
    "curve.title": "Training curve",
    "curve.body":
      "The repository includes a sampled loss curve collected during training.",
    "footer.eyebrow": "Explore the project",
    "footer.title": "Sources and references",
    "footer.docs": "Project Docs",
    "docs.eyebrow": "Project documentation",
    "docs.title": "Siamese Project Docs",
    "docs.subtitle":
      "Architecture, data pipeline, training flow, and repository map.",
    "docs.backHome": "Home",
    "docs.backDocs": "Docs Home",
    "docs.overviewEyebrow": "Overview",
    "docs.overviewTitle": "What this repository implements",
    "docs.overviewBody":
      "This project is a compact PyTorch reimplementation of Siamese Networks for One-Shot Learning. It targets the Omniglot dataset, trains a twin-branch image encoder, and evaluates performance with a 20-way one-shot matching setup.",
    "docs.goalTitle": "Primary goal",
    "docs.goalBody":
      "Learn whether two handwritten character images belong to the same class.",
    "docs.resultTitle": "Reported result",
    "docs.resultBody":
      "The current implementation reports around 89.5% final accuracy, slightly below the original paper's 92%.",
    "docs.mapEyebrow": "Docs map",
    "docs.mapTitle": "Navigate by implementation boundary",
    "docs.repoEyebrow": "Repository map",
    "docs.repoTitle": "What each core file is responsible for",
    "docs.repoModelBody":
      "Defines the Siamese neural network, including the shared convolution tower, the 4096-dimensional projection layer, and the final similarity logit.",
    "docs.repoDatasetBody":
      "Implements the in-memory Omniglot training and testing datasets, pair sampling logic, image rotation augmentation, and one-shot evaluation episode layout.",
    "docs.repoTrainBody":
      "Owns flag parsing, dataloader construction, optimizer setup, training loop, periodic evaluation, checkpoint saving, and final accuracy aggregation.",
    "docs.repoReadmeBody":
      "Provides the original quick-start instructions, requirement list, experiment summary, and the implementation differences relative to the paper.",
    "docsNav.architecture": "Architecture chapter",
    "docsNav.architectureBody":
      "Covers the Siamese encoder structure, embedding path, absolute-difference scoring, and the reasoning behind the final binary logit.",
    "docsNav.dataset": "Dataset chapter",
    "docsNav.datasetBody":
      "Explains how Omniglot samples are loaded, rotated, paired, and assembled into one-shot evaluation episodes.",
    "docsNav.training": "Training chapter",
    "docsNav.trainingBody":
      "Walks through flag parsing, loaders, optimization, checkpointing, test-time precision measurement, and produced artifacts.",
    "docsArch.eyebrow": "Architecture chapter",
    "docsArch.title": "Siamese model architecture",
    "docsArch.subtitle":
      "Shared encoder path, embedding projection, comparison logic, and output scoring.",
    "docsArch.boundaryEyebrow": "Module boundary",
    "docsArch.boundaryTitle": "`model.py` owns the whole similarity function",
    "docsArch.boundaryBody":
      "The repository keeps all model definition logic inside `model.py`. The `Siamese` class exposes `forward_one` for single-branch feature extraction and `forward` for pairwise scoring.",
    "docsArch.encoderEyebrow": "Encoder path",
    "docsArch.encoderTitle": "Shared convolution tower",
    "docsArch.convTitle": "Convolution blocks",
    "docsArch.convBody":
      "The encoder applies four convolution stages with intermittent max pooling. This progressively reduces spatial resolution while expanding channel depth.",
    "docsArch.sharedTitle": "Weight sharing",
    "docsArch.sharedBody":
      "Both input images pass through the same `self.conv` and `self.liner` modules, ensuring that similarity is measured in a common embedding space.",
    "docsArch.forwardEyebrow": "Control flow",
    "docsArch.forwardTitle": "From `forward_one` to final logit",
    "docsArch.step1Title": "1. `forward_one`",
    "docsArch.step1Body":
      "Each branch image is encoded by the convolution stack, flattened, and projected into a 4096-dimensional embedding with sigmoid activation.",
    "docsArch.step2Title": "2. Absolute difference",
    "docsArch.step2Body":
      "`forward` computes `torch.abs(out1 - out2)`, which is the central similarity comparison primitive in this implementation.",
    "docsArch.step3Title": "3. Output layer",
    "docsArch.step3Body":
      "The difference vector is passed to `self.out`, producing a single logit for binary same-class vs different-class classification.",
    "docsArch.tradeoffEyebrow": "Current limitations",
    "docsArch.tradeoffTitle": "Implementation tradeoffs in this repo",
    "docsArch.limit1Title": "Single-file ownership",
    "docsArch.limit1Body":
      "The architecture is compact and easy to read, but the file does not separate encoder, projection, and scoring into independently testable modules.",
    "docsArch.limit2Title": "Minimal output head",
    "docsArch.limit2Body":
      "The model returns logits directly and leaves probability calibration to the loss function and downstream evaluation code.",
    "docsData.eyebrow": "Dataset chapter",
    "docsData.title": "Omniglot data pipeline",
    "docsData.subtitle":
      "In-memory loading, rotation augmentation, pair sampling, and one-shot episodes.",
    "docsData.boundaryEyebrow": "Module boundary",
    "docsData.boundaryTitle": "`mydataset.py` defines both train and test loaders",
    "docsData.boundaryBody":
      "The repository uses two custom dataset classes: `OmniglotTrain` for endless pair sampling during optimization and `OmniglotTest` for one-shot evaluation episodes.",
    "docsData.trainEyebrow": "Training dataset",
    "docsData.trainTitle": "`OmniglotTrain` loads and augments character classes",
    "docsData.memTitle": "In-memory cache",
    "docsData.memBody":
      "`loadToMem` walks the full training tree once and stores PIL images in memory, reducing repeated disk access during long training runs.",
    "docsData.rotateTitle": "Rotation expansion",
    "docsData.rotateBody":
      "Each class is duplicated at 0, 90, 180, and 270 degrees, effectively turning orientation variants into extra class identities.",
    "docsData.samplingEyebrow": "Sampling logic",
    "docsData.samplingTitle": "Positive and negative pair generation",
    "docsData.posTitle": "Positive pairs",
    "docsData.posBody":
      "Odd indices sample two images from the same class and return label `1.0`.",
    "docsData.negTitle": "Negative pairs",
    "docsData.negBody":
      "Even indices sample images from different classes and return label `0.0`.",
    "docsData.transformTitle": "Transforms",
    "docsData.transformBody":
      "The training path applies random affine augmentation before `ToTensor`, which injects mild shape variation into each sampled pair.",
    "docsData.testEyebrow": "Evaluation dataset",
    "docsData.testTitle": "`OmniglotTest` builds one-shot episodes",
    "docsData.episodeTitle": "Episode layout",
    "docsData.episodeBody":
      "Index `0` in each episode creates the anchor image and a true match. The remaining `way - 1` entries are distractor classes.",
    "docsData.metricTitle": "Metric contract",
    "docsData.metricBody":
      "The training loop treats prediction as correct only when the maximum score in the episode lands on the first pair, which is the true match.",
    "docsTrain.eyebrow": "Training chapter",
    "docsTrain.title": "Training loop and runtime behavior",
    "docsTrain.subtitle":
      "Flag parsing, dataloaders, BCEWithLogitsLoss, DataParallel, checkpoints, and test precision.",
    "docsTrain.boundaryEyebrow": "Module boundary",
    "docsTrain.boundaryTitle": "`train.py` owns orchestration end to end",
    "docsTrain.boundaryBody":
      "The script handles flag parsing, path selection, dataloader creation, optimizer setup, checkpointing, intermediate evaluation, and final accuracy reporting.",
    "docsTrain.flagsEyebrow": "Runtime flags",
    "docsTrain.flagsTitle": "Config is driven through `gflags`",
    "docsTrain.pathsTitle": "Path flags",
    "docsTrain.pathsBody":
      "`train_path`, `test_path`, and `model_path` define where Omniglot data is read from and where checkpoints are written.",
    "docsTrain.scheduleTitle": "Schedule flags",
    "docsTrain.scheduleBody":
      "`show_every`, `save_every`, `test_every`, and `max_iter` determine how often loss is printed, checkpoints are saved, and evaluation runs are executed.",
    "docsTrain.optEyebrow": "Optimization",
    "docsTrain.optTitle": "Loss, optimizer, and multi-GPU behavior",
    "docsTrain.lossTitle": "BCEWithLogitsLoss",
    "docsTrain.lossBody":
      "The model emits raw logits and the script applies `torch.nn.BCEWithLogitsLoss` directly, which keeps the sigmoid inside the numerically stable loss function.",
    "docsTrain.adamTitle": "Adam optimizer",
    "docsTrain.adamBody":
      "The training loop uses Adam rather than SGD with momentum, which is one of the documented reasons the final metric differs from the paper.",
    "docsTrain.gpuTitle": "DataParallel",
    "docsTrain.gpuBody":
      "If multiple GPU ids are provided, the script wraps the network with `torch.nn.DataParallel` after constructing the Siamese model.",
    "docsTrain.evalEyebrow": "Evaluation and artifacts",
    "docsTrain.evalTitle": "How the script measures precision and saves state",
    "docsTrain.precisionTitle": "Episode precision",
    "docsTrain.precisionBody":
      "For each test episode, the model compares all candidates and treats the prediction as correct only if `np.argmax(output)` returns the first item.",
    "docsTrain.ckptTitle": "Checkpointing",
    "docsTrain.ckptBody":
      "The script saves intermediate weights under names like `model-inter-<step>.pt` and stores sampled loss history into a `train_loss` pickle file.",
  },
  zh: {
    "brand.eyebrow": "孪生网络 · Omniglot · PyTorch",
    "brand.title": "用于单样本学习的孪生网络",
    "brand.subtitle": "基于 Omniglot 数据集的 PyTorch 复现",
    "nav.docs": "文档",
    "hero.eyebrow": "单样本学习系统",
    "hero.title": "双塔编码，一次比对得出判断。",
    "hero.body":
      "这是一个在 Omniglot 上训练的紧凑型 Siamese PyTorch 实现，把论文思路直接落到可运行代码。",
    "hero.ctaRepo": "GitHub 仓库",
    "hero.ctaDocs": "项目文档",
    "hero.ctaQuickStart": "快速开始",
    "metrics.label": "关键指标",
    "metrics.accuracy": "最终准确率",
    "metrics.evaluation": "20 路单样本测试",
    "metrics.framework": "实现框架",
    "metrics.dataset": "数据集",
    "architecture.label": "结构速览",
    "architecture.note": "共享双塔编码器，取绝对差值，再经过最终线性层输出分数。",
    "quickStart.eyebrow": "运行项目",
    "quickStart.title": "快速开始",
    "quickStart.body":
      "准备 Omniglot 数据、创建模型目录，然后直接运行仓库自带的训练脚本。",
    "notes.eyebrow": "实现说明",
    "notes.title": "结果为何低于论文",
    "notes.optimizerTitle": "优化器选择",
    "notes.optimizerBody": "当前实现使用 Adam，而不是带动量的 SGD。",
    "notes.paramsTitle": "参数设置",
    "notes.paramsBody":
      "代码沿用了 PyTorch 默认初始化和统一配置，而没有复现论文中的分层参数策略。",
    "curve.eyebrow": "实验产物",
    "curve.title": "训练曲线",
    "curve.body": "仓库内包含一次训练过程中按批次采样得到的 loss 曲线。",
    "footer.eyebrow": "继续查看",
    "footer.title": "源码与参考资料",
    "footer.docs": "项目文档",
    "docs.eyebrow": "项目文档",
    "docs.title": "Siamese 项目详解",
    "docs.subtitle": "包含架构、数据管线、训练流程与仓库文件说明。",
    "docs.backHome": "返回首页",
    "docs.backDocs": "文档首页",
    "docs.overviewEyebrow": "整体概览",
    "docs.overviewTitle": "这个仓库具体实现了什么",
    "docs.overviewBody":
      "这个项目是 Siamese Networks for One-Shot Learning 的一个紧凑型 PyTorch 复现版本。它使用 Omniglot 数据集，训练双分支图像编码器，并通过 20-way one-shot 匹配任务评估性能。",
    "docs.goalTitle": "核心目标",
    "docs.goalBody": "判断两张手写字符图像是否属于同一个类别。",
    "docs.resultTitle": "当前结果",
    "docs.resultBody":
      "当前实现报告的最终准确率约为 89.5%，略低于原论文中的 92%。",
    "docs.mapEyebrow": "文档地图",
    "docs.mapTitle": "按实现边界导航",
    "docs.repoEyebrow": "仓库地图",
    "docs.repoTitle": "核心文件分别负责什么",
    "docs.repoModelBody":
      "定义 Siamese 神经网络，包括共享卷积编码器、4096 维投影层以及最终的相似度 logit 输出。",
    "docs.repoDatasetBody":
      "实现 Omniglot 训练集与测试集的内存加载、样本对采样、旋转增强，以及 one-shot episode 组织逻辑。",
    "docs.repoTrainBody":
      "负责参数解析、dataloader 构建、优化器初始化、训练循环、周期性评估、checkpoint 保存和最终准确率汇总。",
    "docs.repoReadmeBody":
      "提供原始快速开始说明、依赖要求、实验结果摘要，以及与论文实现差异的说明。",
    "docsNav.architecture": "架构章节",
    "docsNav.architectureBody":
      "解释 Siamese 编码器结构、嵌入路径、绝对差比较方式，以及最终二分类 logit 的来源。",
    "docsNav.dataset": "数据章节",
    "docsNav.datasetBody":
      "说明 Omniglot 样本如何被加载、旋转、配对，并组织成 one-shot 评估 episode。",
    "docsNav.training": "训练章节",
    "docsNav.trainingBody":
      "说明参数解析、dataloader、优化过程、checkpoint、测试精度计算和训练产物。",
    "docsArch.eyebrow": "架构章节",
    "docsArch.title": "Siamese 模型架构",
    "docsArch.subtitle": "覆盖共享编码路径、嵌入投影、比较逻辑和最终打分。",
    "docsArch.boundaryEyebrow": "模块边界",
    "docsArch.boundaryTitle": "`model.py` 负责完整的相似度函数",
    "docsArch.boundaryBody":
      "仓库把模型定义逻辑全部放在 `model.py` 中。`Siamese` 类通过 `forward_one` 负责单分支特征提取，通过 `forward` 负责成对样本打分。",
    "docsArch.encoderEyebrow": "编码路径",
    "docsArch.encoderTitle": "共享卷积塔",
    "docsArch.convTitle": "卷积模块",
    "docsArch.convBody":
      "编码器包含四个卷积阶段，并穿插最大池化，逐步降低空间分辨率并提升通道数。",
    "docsArch.sharedTitle": "参数共享",
    "docsArch.sharedBody":
      "两张输入图片都会经过同一套 `self.conv` 和 `self.liner`，保证相似度比较发生在统一的嵌入空间里。",
    "docsArch.forwardEyebrow": "控制流",
    "docsArch.forwardTitle": "从 `forward_one` 到最终 logit",
    "docsArch.step1Title": "1. `forward_one`",
    "docsArch.step1Body":
      "每个分支输入先通过卷积堆栈，再展平，并经由线性层加 sigmoid 投影到 4096 维嵌入空间。",
    "docsArch.step2Title": "2. 绝对差比较",
    "docsArch.step2Body":
      "`forward` 中通过 `torch.abs(out1 - out2)` 计算两个嵌入向量的逐元素绝对差，这是这份实现的核心比较方式。",
    "docsArch.step3Title": "3. 输出层",
    "docsArch.step3Body":
      "差异向量会送入 `self.out`，输出一个单独的 logit，用于判断同类或异类。",
    "docsArch.tradeoffEyebrow": "当前限制",
    "docsArch.tradeoffTitle": "这份实现里的取舍",
    "docsArch.limit1Title": "单文件承载",
    "docsArch.limit1Body":
      "当前结构紧凑、容易读懂，但没有把编码器、投影层和打分头拆成可以独立测试的模块。",
    "docsArch.limit2Title": "输出头最小化",
    "docsArch.limit2Body":
      "模型直接返回 logits，把概率校准交给 loss 函数和后续评估逻辑处理。",
    "docsData.eyebrow": "数据章节",
    "docsData.title": "Omniglot 数据管线",
    "docsData.subtitle": "覆盖内存加载、旋转增强、样本对采样与 one-shot episode 组织。",
    "docsData.boundaryEyebrow": "模块边界",
    "docsData.boundaryTitle": "`mydataset.py` 同时定义训练集和测试集",
    "docsData.boundaryBody":
      "仓库使用两个自定义数据集类：`OmniglotTrain` 用于训练阶段的连续样本对采样，`OmniglotTest` 用于 one-shot 评估 episode 构造。",
    "docsData.trainEyebrow": "训练集",
    "docsData.trainTitle": "`OmniglotTrain` 负责加载并增强字符类别",
    "docsData.memTitle": "内存缓存",
    "docsData.memBody":
      "`loadToMem` 会遍历完整训练目录并把 PIL 图像缓存到内存中，从而减少长时间训练中的重复磁盘读取。",
    "docsData.rotateTitle": "旋转扩充",
    "docsData.rotateBody":
      "每个类别都会扩展出 0、90、180、270 度四种旋转版本，相当于把朝向变化当作额外类别处理。",
    "docsData.samplingEyebrow": "采样逻辑",
    "docsData.samplingTitle": "正负样本对如何生成",
    "docsData.posTitle": "正样本对",
    "docsData.posBody":
      "奇数索引会从同一个类别里采样两张图像，并返回标签 `1.0`。",
    "docsData.negTitle": "负样本对",
    "docsData.negBody":
      "偶数索引会从不同类别中采样图像，并返回标签 `0.0`。",
    "docsData.transformTitle": "图像变换",
    "docsData.transformBody":
      "训练路径会在 `ToTensor` 之前施加随机仿射增强，为样本对引入轻微形变扰动。",
    "docsData.testEyebrow": "测试集",
    "docsData.testTitle": "`OmniglotTest` 如何组织 one-shot episode",
    "docsData.episodeTitle": "Episode 结构",
    "docsData.episodeBody":
      "每个 episode 的索引 `0` 会生成 anchor 图像及其真实匹配项，其余 `way - 1` 个位置则为干扰类别。",
    "docsData.metricTitle": "评估契约",
    "docsData.metricBody":
      "训练脚本只有在一个 episode 内最高得分落在第一个样本对上时，才认为预测成功。",
    "docsTrain.eyebrow": "训练章节",
    "docsTrain.title": "训练循环与运行时行为",
    "docsTrain.subtitle":
      "覆盖参数解析、dataloader、BCEWithLogitsLoss、DataParallel、checkpoint 与测试精度。",
    "docsTrain.boundaryEyebrow": "模块边界",
    "docsTrain.boundaryTitle": "`train.py` 负责端到端调度",
    "docsTrain.boundaryBody":
      "这个脚本处理参数解析、路径选择、dataloader 构建、优化器初始化、checkpoint 保存、中途评估和最终准确率输出。",
    "docsTrain.flagsEyebrow": "运行参数",
    "docsTrain.flagsTitle": "配置通过 `gflags` 驱动",
    "docsTrain.pathsTitle": "路径参数",
    "docsTrain.pathsBody":
      "`train_path`、`test_path` 和 `model_path` 决定了 Omniglot 数据读取位置与 checkpoint 输出位置。",
    "docsTrain.scheduleTitle": "调度参数",
    "docsTrain.scheduleBody":
      "`show_every`、`save_every`、`test_every` 和 `max_iter` 控制 loss 输出、checkpoint 保存和评估执行频率。",
    "docsTrain.optEyebrow": "优化过程",
    "docsTrain.optTitle": "Loss、优化器与多卡行为",
    "docsTrain.lossTitle": "BCEWithLogitsLoss",
    "docsTrain.lossBody":
      "模型输出原始 logits，脚本直接使用 `torch.nn.BCEWithLogitsLoss`，把 sigmoid 留在数值更稳定的损失函数内部。",
    "docsTrain.adamTitle": "Adam 优化器",
    "docsTrain.adamBody":
      "训练过程默认使用 Adam，而不是带动量的 SGD，这也是结果低于论文的已知原因之一。",
    "docsTrain.gpuTitle": "DataParallel",
    "docsTrain.gpuBody":
      "如果提供多个 GPU id，脚本会在构建 Siamese 模型后通过 `torch.nn.DataParallel` 包装网络。",
    "docsTrain.evalEyebrow": "评估与产物",
    "docsTrain.evalTitle": "脚本如何计算精度并保存状态",
    "docsTrain.precisionTitle": "Episode 精度",
    "docsTrain.precisionBody":
      "对每个测试 episode，模型会比较所有候选项，只有当 `np.argmax(output)` 返回第一个位置时才判定为正确。",
    "docsTrain.ckptTitle": "Checkpoint 保存",
    "docsTrain.ckptBody":
      "脚本会保存 `model-inter-<step>.pt` 这类中间权重文件，并把采样得到的 loss 历史写入 `train_loss` pickle 文件。",
  },
};

function getInitialLanguage() {
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored === "zh" ? "zh" : DEFAULT_LANGUAGE;
}

function applyLanguage(language) {
  const nextLanguage = language === "zh" ? "zh" : DEFAULT_LANGUAGE;
  const copy = translations[nextLanguage];

  document.documentElement.lang = nextLanguage === "zh" ? "zh-CN" : "en";

  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.dataset.i18n;
    if (copy[key]) {
      node.textContent = copy[key];
    }
  });

  document.querySelectorAll("[data-lang]").forEach((button) => {
    const active = button.dataset.lang === nextLanguage;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  });

  localStorage.setItem(STORAGE_KEY, nextLanguage);
}

document.querySelectorAll("[data-lang]").forEach((button) => {
  button.addEventListener("click", () => applyLanguage(button.dataset.lang));
});

applyLanguage(getInitialLanguage());
