# Code Agent 流体渲染结果静态 Review（第 2 版）

> 第 2 版相对第 1 版的变化：新增 `deepseel_v4_pro_debug`（DeepSeek 人工 debug 后的版本）；**新增对参考实现 `main` 的同标准评审（第 11 节）与四方对比（第 12 节）**；逐行复核了三条管线的深度约定、滤波语义、绑定布局和参数量级；修正了第 1 版中三处结论（见第 3 节）。本版所有引用均带文件与行号，便于复核。
>
> 第 1–10 节只涉及三个模型的产出（评审时未参考 `main`，以避免以参考实现为文本相似度基准）。第 11–12 节是把同一套标准施加到 `main` 上的结果，用于校准评分刻度。

## 1. Review 范围与方法

| 编号 | 模型/版本 | 结果目录 | 说明 |
| --- | --- | --- | --- |
| A | Opus 5 | `bench_results/opus_5` | 模型直接输出 |
| B | GPT 5.6 Sol Medium | `bench_results/gpt_5_6_sol_medium` | 模型直接输出 |
| C | DeepSeek V4 Pro | `bench_results/deepseek_v4_pro` | 模型直接输出 |
| C' | DeepSeek V4 Pro (debug) | `bench_results/deepseel_v4_pro_debug` | 在 C 基础上人工 debug 后的版本 |

评审基准：测试起点 `benchmark/raw-particles-baseline`（即当前工作树 `src/`）、`NarrowRangeFilter.pdf` 的方法语义、Benchmark Prompt 的三项任务。

**方法边界**：本轮只做静态代码阅读，未安装依赖、未执行 `npm run build`、未编译 WGSL、未启动浏览器、未观察画面、未 profiling。因此：

- "完成" = 代码在功能设计与数学推导上覆盖了任务要求；
- 不等于通过 WebGPU validation，不等于画面正确，不等于性能达标；
- 文中所有对画面的判断均标注为**静态推断**，并在第 8 节给出对应的最小验证方法。

### 1.1 改动范围（已排除 CRLF/LF 行尾差异）

| 版本 | 改动文件 | 新增目录 |
| --- | --- | --- |
| A Opus | `common/config.ts`、`common/shader.ts`、`controller.ts`、`renderer/rawParticles/{particles,shader}.ts`、`simulator/SPH.ts` | `renderer/fluid/`（4 文件） |
| B GPT | `controller.ts`、`renderer/rawParticles/{particles,shader}.ts` | 无 |
| C/C' DeepSeek | `controller.ts` | `renderer/fluid/`（4 文件） |

B 的 `controller.ts` 只增加了 2 处 `prepare()` 调用（共 6 行），是三者中侵入最小的集成方式。C/C' 完全替换了粒子渲染路径，`renderer/rawParticles/` 变为未被引用的死代码。A 保留了原始粒子视图并把它也改成了 imposter。

### 1.2 基线的深度与相机约定（三者共同前提）

复核 `src/controller.ts:15-25` 与 `src/renderer/globalResource.ts:94-111`：

- 投影矩阵被 patch 成 **Reverse-Z**（`depthClearValue: 0.0` + `depthCompare: 'greater'`，见 `src/controller.ts:385-390`）；
- `camera.params = (aspect·h/n, −h/n, (f−n)/(n·f), 1/f)`，满足 `eyeDepth = 1/(params.z·d + params.w)`，代入 `d=1` 得 `near`、`d=0` 得 `far`，确认 Reverse-Z；
- `params.y` 为负，已内含屏幕 Y 向下与 view 空间 Y 向上的翻转。

三份实现对这套约定的理解都正确，但表达方式不同：

| 版本 | 流体深度纹理存什么 | 无流体哨兵 | 反投影方式 |
| --- | --- | --- | --- |
| A Opus | eye space z（负值） | `0.0` | `camera.params.xy`（复用基线约定） |
| B GPT | eye space z（负值） | `−1e6`（clear）/ `−1e5`（比较阈值） | `projectionMatrix[0][0]`、`[1][1]` |
| C/C' DeepSeek | Reverse-Z NDC 深度 | `0.0` | `invProj` 矩阵（额外传入） |

A/B 在**线性 eye 深度**上做滤波，C/C' 在**非线性 NDC 深度**上做滤波。后者导致同一个 `sigmaRange` 在近处和远处含义完全不同，是 C/C' 滤波质量问题的根源之一。

## 2. 总体结论

| 版本 | 深度滤波 | Screen-space 流体 | Billboard | 厚度 | 静态完成状态 |
| --- | --- | --- | --- | --- | --- |
| A Opus 5 | **论文语义完整**（Eq.2/3/5/6/8/9 + 2D clean-up） | 完成 | 4 顶点 strip | 有 | **成功，当前最佳** |
| B GPT 5.6 Sol Medium | **论文语义完整**（同上，边界处理略逊） | 完成 | 6 顶点 list | 有 | **成功** |
| C DeepSeek V4 Pro | 双边滤波，非论文方法 | 框架完成 | 6 顶点 list | 无 | **失败（无法运行）** |
| C' DeepSeek (debug) | 双边滤波，非论文方法 | 框架完成 | 6 顶点 list | 无 | **部分完成（可运行，但存在高风险画面缺陷）** |

静态排序：

```text
Opus 5  >  GPT 5.6 Sol Medium  >>  DeepSeek (debug)  >  DeepSeek (原始)
```

A 与 B 的差距不在"是否实现"，而在边界语义、资源设计和可验证性；A 与 B 对 C/C' 的差距是"是否实现了论文算法"这一量级的差别。

## 3. 对第 1 版结论的修正

第 1 版有三处判断需要修正，均为对 Opus 不利或对差异描述不准确的地方：

**3.1 撤回"Opus billboard 投影误差"这一保留项。**
第 1 版指出 Opus 用切锥半径扩大 quad（`common/shader.ts:96-100`：`halfExtent = r·d/√(d²−r²)`）而 fragment 用正交近似，存在位置与深度不一致。数学上成立，但量级被夸大：本项目 `r = 0.012`、相机距离约 `2.83`，`d/r ≈ 236`，`halfExtent/r = 1 + 1/(2·236²) ≈ 1.000009`，误差在 float 精度以下。该项应从缺陷清单移除。B 采用 `center + local·r`（球内切于 quad），在极近处会被自身 quad 裁剪；两者在本项目参数下都无可观察差异，不构成区分点。

**3.2 修正"Opus 最大核半径 48 像素"。**
`MAX_KERNEL_RADIUS = 48`（`fluid/fluidRenderer.ts:41`）只是 override 常量的硬上限，实际半径为 `min(3σ, 48)`，而 `σ ≤ maxFilterSigma = 12`（`common/config.ts:32`），故上限是 36 像素。按默认参数代入 Eq.5 的典型值只有 18 像素（见第 7.2 节）。原描述会让读者误以为默认开销远高于实际。

**3.3 新增第 1 版未发现的 GPT 边界缺陷。**
GPT 的滤波把**无流体的背景样本也当作"过远样本"钳制成 `zi − mu` 混入加权**（`rawParticles/shader.ts:169-170`，`isFluid(zj)` 为假时 `select` 落到 `zi − mu` 分支）。Opus 则在遇到背景时终止核（`fluid/filterShader.ts:130` `break`）。论文的钳制对象是"属于同一表面但超出允许深度范围"的样本，背景应当终止核而不是参与加权。这是两者在流体轮廓处的实质差异，第 1 版未识别。

## 4. A — Opus 5

文件：`renderer/fluid/{particleShader,filterShader,compositeShader,fluidRenderer}.ts`，`common/shader.ts`（新增 `FluidOptions`/`SphereImposter`），`common/config.ts`（新增 `renderOptions`）。

### 4.1 管线

```text
主 pass（mesh + skybox → canvas, renderDepthMap）
  → thickness pass   imposter 加法混合 → r16float，depthReadOnly 用场景深度剔除
  → depth pass       imposter → r32float(eye z) + 写 renderDepthMap
  → filter           1 个 compute pass 内 4 次 1D dispatch + 1 次 5×5 clean-up，ping-pong
  → composite        全屏三角形，alpha 混合覆盖到 canvas
```

### 4.2 正确实现的关键点

1. **4 顶点 triangle-strip**，corner 由 `vertexIndex` 位运算得出（`common/shader.ts:87-92`），三者中顶点量最低。
2. **thickness pass 先于 depth pass，且 `depthReadOnly: true`**（`fluidRenderer.ts:336`）。注释（`fluidRenderer.ts:18-23`）明确解释了原因：厚度必须累加表面之后的所有粒子，因此不能写深度、不能自遮挡，但仍需用场景深度剔除被实体挡住的粒子。**这是三者中唯一正确处理"厚度累加"与"深度剔除"关系的实现。**
3. **场景遮挡在光栅化阶段解决**：两个 imposter pass 都以 `renderDepthMap` 为深度附件（`fluidRenderer.ts:336`、`356-360`），被实体遮挡的粒子从不进入深度图，因此不会把表面拖过遮挡边界。composite 因此无需深度测试，直接 `discard` 无流体像素。
4. **Narrow-Range Filter 语义完整**（`fluid/filterShader.ts`）：
   - Eq.5 深度相关核尺寸：`σ = H·filterSigma/(2|z|·tan(α/2))`，`clamp(σ,1,maxFilterSigma)`，`radius = min(3σ, 48)`（48-54、115-116）；
   - Eq.2 过远样本钳制到 `center − mu`（72-73）；
   - Eq.3/6 成对取舍——一对中任一侧属于前景则整对丢弃，维持核对称（66-68）；
   - Eq.8/9 动态范围扩展，由近及远逐步放宽（79-86）；
   - **背景终止核**（130 `break`）与**屏幕外视作背景**（42），边界语义与论文一致；
   - 论文 3.4 节的 5×5 2D clean-up pass，复用同一 `accumulatePair`，且用 `dy==0 && dx<=0 continue` 正确遍历半核避免重复计数（172-191）。
5. **pipeline override constants** 生成 H/V 两条管线（`fluidRenderer.ts:254-263`），共享一个 shader module，方向判断零运行时开销。
6. **法线重建最讲究**（`compositeShader.ts:44-76`）：两侧都有流体时选择**深度变化较小**的一侧做单边差分，避免法线跨断层被拉扯；只有一侧有流体时退化为单边差分；`lengthSqr < 1e-24` 时返回 `(0,0,1)`，无 NaN 风险。
7. **着色的物理取舍有依据**（`compositeShader.ts:112-147`）：
   - 薄层折射方向按 `thickness/(8r)` 渐变回视线方向，避免薄水花采到无关的暗区；
   - Beer-Lambert 吸收 + 一个单散射项（用 envMap 上方向作 ambient），避免厚水体直接变黑；
   - specular 被同一 Fresnel 项调制而非叠加，避免波纹爆白；
   - `alpha = 1 − exp(−opacity·thickness)`，薄层/水花与背景 alpha 混合——三者中唯一做混合的；
   - 复用基线 `sRGBGammaEncode`，与场景其余部分颜色空间一致。
8. **uniform 布局手算正确**：`FluidOptions` 8 个 f32（0–31）+ `vec3<f32>`（对齐到 32）+ 2 个 f32（44、48），结构 size 向上对齐到 64，与 `OPTIONS_BUFFER_SIZE = 64` 完全一致（`fluidRenderer.ts:39`、`144-154`）。这正是 C 出错的地方。
9. **compute pass 内 ping-pong 合法**：WebGPU 的 compute pass usage scope 是 per-dispatch，同一 pass 内交替读写两张纹理是允许的，因此 5 次 dispatch 合并进 1 个 pass（`fluidRenderer.ts:376-395`），比 B 的 5 个独立 pass 少 4 次 pass 开销。
10. **原始粒子视图也改成了 imposter**（`rawParticles/shader.ts`、`particles.ts`），删除了 `SphereGeometry(0.007,8,8)` 和全部顶点/索引缓冲，与流体路径共享 `SphereImposter` 代码，无重复实现。对"改成更高效的渲染方式"这一要求响应最彻底。
11. **完整可调 UI**（`common/config.ts:22-36`、`73-88`）：渲染模式切换、imposter 放大倍数、迭代次数、clean-up 开关、σ/δ/μ、核上限、IOR、吸收、不透明度、流体颜色。滤波参数以 imposter 半径的倍数表达，默认 `σ=0.7r, δ=10r, μ=r`，即论文推荐值。

### 4.3 保留项

1. **alpha 混合发生在 gamma 编码空间**：shader 输出 `sRGBGammaEncode(color)` 且 `alpha < 1`，固定功能混合在非线性空间进行（`fluidRenderer.ts:303-306`），半透明边缘会偏亮。数学上不正确，但这是实时渲染的普遍做法，视觉影响有限。
2. **depth pass 使用 `depthStoreOp: 'store'`**（`fluidRenderer.ts:359`），会把粒子深度写回 `renderDepthMap`。当前无害（composite 不读它，下一帧会 clear），但 `'discard'` 语义更干净，也能省一次写回带宽。
3. **`camera.fov` 只在 init 时采样一次**（`fluidRenderer.ts:128`），运行时改 FOV 不会更新 `tanHalfFov`。基线未暴露 FOV 控制，当前不触发。
4. **无窗口 resize 处理**：纹理尺寸在 init 时固定。三份实现均如此，属于基线遗留而非本次回归。
5. **折射只查 cubemap，看不到水中的场景物体**：没有对 `sceneColor` 做屏幕空间折射。A/B/C' 都是如此，是共同的简化。

### 4.4 静态结论

三项任务全部完成，且在论文语义、WebGPU 资源设计、颜色空间、可验证性四个方面都是最完整的。代码注释直接引用论文公式编号，对 `discard` 后 helper invocation 仍会执行、r32float 不可过滤、compute pass usage scope 等 GPU/API 细节的处理体现了确实的理解而非模仿。

## 5. B — GPT 5.6 Sol Medium

文件：`renderer/rawParticles/{shader,particles}.ts`（原地替换球体渲染），`controller.ts`（+6 行）。

### 5.1 管线

```text
prepare()：depth pass → thickness pass → H1 → V1 → H2 → V2 → cleanup（5 个独立 compute pass）
主 pass：mesh → skybox → composite（全屏三角形，写 frag_depth，与场景共享 renderDepthMap 做深度测试）
```

### 5.2 正确实现的关键点

1. **Narrow-Range Filter 语义完整**（`shader.ts:136-177`）：Eq.5 深度相关核（`σ = H·worldSigma·proj[1][1]/(2|zi|)`，141-146）、动态范围扩展（157-162）、前景成对剔除（164-166）、过远钳制（169-170）、对称加权（171-172），并实现了 5×5 clean-up（196-236）。
2. **ping-pong 链条闭合**：`raw→A, A→B, B→A` 三个 bind group（`particles.ts:143-147`），按 `bg0→bg1→bg2→bg1→bg2` 执行（238-242），最终结果落在 `filteredDepthA`，与 composite 读取的纹理一致（178）。链条正确。
3. **厚度物理正确**：`2r√(1−d²)` 弦长 + 加法混合（`shader.ts:107`、`particles.ts:116-119`）。
4. **composite 写 `frag_depth` 并参与主 pass 深度测试**（`shader.ts:333`、`particles.ts:194`），流体被场景实体正确遮挡，且与基线单 pass 结构无缝集成——`controller.ts` 只多了 2 行 `prepare()` 调用，是三者中最干净的集成。
5. **法线重建无退化风险**：断层处邻居退化为中心深度，但反投影使用的是邻居**像素坐标**，因此差分向量的 xy 分量仍非零（`shader.ts:295-305`），不会出现 `normalize(0)`。
6. **着色数学正确**：`refract(I,N,1/1.333)` + 全反射回退（314-315）、Fresnel `F0 = 0.02037`（325，水的准确值）、Beer-Lambert 逐通道吸收 `exp(−(5.0,1.4,0.55)·t)` + 水体色补偿（320-322）、`gammaEncode(1/2.2)`（284-286）。
7. **滤波参数直接采用论文推荐值**：`σ=0.7r, δ=10r, μ=r`（`particles.ts:68-76`，注释明确说明），并把 imposter 半径放大到 `0.009`（1.5×）以填补空隙，注释交代了动机。
8. **`@interpolate(flat)` 传递粒子中心**（`shader.ts:24`），避免不必要的透视插值。

### 5.3 主要问题

**5.3.1 背景样本被当作"过远样本"混入加权（新发现）**

`shader.ts:169-170`：

```wgsl
let fj = select(zi - filterParams.mu, zj, isFluid(zj) && zj >= zi - deltaLow);
```

`zj` 为背景（`−1e6`）时 `isFluid` 为假，落入 `zi − mu` 分支被计入加权；同时 `foregroundJ` 也因 `isFluid` 为假而不成立，所以整对不会被丢弃。结果是流体轮廓附近半个核都是 `zi − mu` 样本，边缘深度被系统性拉远约 `μ = r`。Opus 在同一位置选择 `break` 终止核。

静态推断的画面影响：流体轮廓会出现一圈约一个粒子半径深的下沉/后倾斜坡。它可能在观感上误以为是"边缘曲率"，但与论文的边缘曲率来源不同——论文的曲率来自同一表面上被钳制的远端样本，而不是来自背景。

**5.3.2 屏幕边界按 clamp 到边缘处理**

`loadDepth` 把越界坐标 `clamp` 到边缘像素（`shader.ts:133`），等于把边缘像素复制到屏幕外，屏幕四边的流体会被额外平滑拉伸。Opus 把越界视作背景（`filterShader.ts:42`），语义更正确。

**5.3.3 场景预遮挡缺失**

depth pass 使用私有的 `particleDepthTexture`（`particles.ts:55-58`、`216-219`，`depthClearValue: 0.0` 且不 load 场景深度），thickness pass 完全没有深度附件（226-232）。因此被实体挡住的粒子仍然进入深度图与厚度累积，会影响：

- 物体轮廓附近的滤波深度与由差分得到的法线；
- 流体厚度与吸收（厚度偏大）。

最终 composite 的深度测试能阻止流体覆盖实体，所以不会出现"流体盖住物体"的硬错误，属于次级质量问题。Boundary 场景与 Water Droplet 场景最容易暴露。

**5.3.4 固定 20 次循环**

`for (var step = 1; step <= MAX_RADIUS; step++) { if (step <= radius) {...} }`（`shader.ts:152-153`）：无论实际 `radius` 多小，循环体始终执行 20 次。保证了均匀控制流，但按默认参数实际 radius 约 9（见 7.2），有一半以上迭代是空转。Opus 用 `break` 提前退出。

**5.3.5 工程可验证性弱**

- 参数硬编码在 `particles.ts:70`、`76` 的两个 `Float32Array` 字面量里，无 UI；
- 没有原始深度/滤波深度/法线/厚度的调试显示模式；
- 没有保留原始粒子视图（`RawParticles` 被原地替换），无法对照；
- 固定 4 次 1D pass + 1 次 clean-up，无运行时质量/性能取舍。

**5.3.6 次级资源问题**

- 厚度用 `rgba16float`（`particles.ts:59-62`），只用到 R 通道，显存是 `r16float` 的 4 倍；
- `prepare()` 每帧调用 `createView()` 创建纹理视图（213、217、229），每帧产生一批临时对象。A 与 C' 都缓存了 view。

### 5.4 静态结论

三项任务全部完成，论文滤波与最终着色都扎实，集成方式最克制。与 A 的差距集中在滤波边界语义（5.3.1/5.3.2）、场景预遮挡（5.3.3）、循环效率（5.3.4）和可验证性（5.3.5），而不是核心功能缺失。

## 6. C / C' — DeepSeek V4 Pro（原始版与 debug 版）

### 6.1 debug 版修复了什么

`diff -ru deepseek_v4_pro/src deepseel_v4_pro_debug/src` 显示 8 处修复，与第 1 版 review 指出的问题基本对应：

| 第 1 版问题 | debug 版处理 | 结论 |
| --- | --- | --- |
| #1 `texture_depth_2d` 的 `textureSampleLevel` level 类型 | 改为 `textureLoad(..., vec2<i32>(input.position.xy), 0)`（`surfaceShader.ts:90`） | **已修**，且改用 framebuffer 坐标比 UV 采样更稳 |
| #2 场景深度 sample type 不匹配 | `unfilterable-float` → `depth`（`fluid.ts:209`） | **已修**（这是原版会让 `createBindGroup` 直接抛错的项） |
| #3 `FilterParams` uniform 尺寸不足 | 补 `_pad: f32`，CPU 端 6 floats = 24 字节（`filterShader.ts:37`、`fluid.ts:124-127`） | **已修** |
| #4 H/V 共享被覆盖的 uniform buffer | 拆成 `filterHParamsBuffer` / `filterVParamsBuffer`，方向烧死在各自 buffer 里（`fluid.ts:126-127`、`250`、`260`） | **已修**，且去掉了每帧 `writeBuffer` |
| #5 billboard 未使用场景深度 | 改用 `sceneDepthView` + `depthLoadOp: 'load'`（`fluid.ts:314-317`） | **修复方式引入了新缺陷，见 6.3** |
| #6 法线朝向反向风险 | `normalize(viewPos)` → `normalize(-viewPos)`（`surfaceShader.ts:75`） | **已修** |
| #7 全屏三角形 UV 垂直翻转 | UV 由 `(0,2),(0,0),(2,0)` 改为 `(0,-1),(0,1),(2,1)`（`filterShader.ts:16-19`、`surfaceShader.ts:17-19`） | **已修**，验算：clip `y=−1`（屏幕下）↔ `v=1`（纹理下），插值正确 |
| #8 无粒子 fallback pass 缺深度附件 | 新增 `beginCanvasPass()` 带 depth attachment（`fluid.ts:346-361`） | **已修** |

另外补上了 `ShaderStruct.DirectionalLight`（`GlobalGroup` 声明 `light: DirectionalLight` 需要该结构定义，原版缺失会导致 WGSL 编译失败），并把 `var<storage>` 显式写成 `var<storage, read>`。

**因此：C（原始输出）的状态应从第 1 版的"部分完成"下调为"失败"** —— 缺少 `DirectionalLight` 结构定义会让 billboard/surface 的 WGSL 编译失败，`depth` sample type 不匹配会让 bind group 创建抛错，二者都在页面初始化阶段即中断。C 不具备可运行性。

### 6.2 debug 版未修复的部分

**6.2.1 滤波仍不是 Narrow-Range Filter**

`filterShader.ts:80-83` 仍是标准可分离双边高斯：

```wgsl
let spatialW = exp(-offset * offset / twoSigmaS2);
let rangeW = exp(-depthDiff * depthDiff / twoSigmaR2);
let w = spatialW * rangeW;
```

缺少论文区别于双边滤波的全部四项关键行为：过远样本钳制、成对偏差修正、动态深度范围扩展、深度相关屏幕核尺寸。`sigmaSpatial` 硬编码 2.0（53），`kernelRadius = 4`、`sigmaRange = 0.001` 来自 uniform 但无 UI（`fluid.ts:124-125`）。**任务 1 不能判为完成。**

此外该滤波作用在**非线性 NDC 深度**上，固定 `sigmaRange` 在近处与远处的物理含义不同；`centerDepth <= 0.0001` 直接返回（49-51），滤波不会扩张流体区域，粒子圆之间的空洞无法被填补。

**6.2.2 仍无厚度/体积近似**

没有 thickness pass。折射与反射可工作，但透射不随流体厚度变化，薄水层、厚水体与飞溅缺乏可区分的吸收，也没有 alpha 混合。

**6.2.3 颜色空间不一致**

`surfaceShader.ts:134` 直接输出 `fluidColor + spec`，未调用基线的 `sRGBGammaEncode`。envMap 经 `rgba8unorm-srgb` view 采样得到**线性**值，而同一画面里的 `sceneColor` 来自 mesh/skybox shader，是**已 gamma 编码**的值。两者混在一张 canvas 上，静态推断流体区域会明显偏暗，且与背景不连续。

**6.2.4 法线存在 NaN 风险**

`surfaceShader.ts:64-72`：邻居无流体时 fallback 用**中心的 view position 本身**，两侧同时 fallback 时 `dp_dx = viewPos − viewPos = 0`，`normalize(cross(0, ...))` 产生 NaN。孤立的单像素/细条流体会出现异常亮点或黑点。B 在同一位置回退到邻居**像素坐标**，不会退化。

**6.2.5 billboard 半径未放大**

`fluid.ts:48` 默认 `particleRadius = 0.006`，与模拟半径相同，粒子投影之间近似相切。配合 6.2.1 的"不扩张流体区域"，圆之间必然留有空洞。A 用 2.0×、B 用 1.5×。

**6.2.6 工程可验证性最弱**

无渲染参数 UI、无中间缓冲调试显示、无原始粒子视图（`rawParticles/` 成为未被引用的死代码）。为 H/V 创建了两条**完全相同**的 pipeline（`fluid.ts:200-201`，仅 label 不同），是无意义的冗余。

### 6.3 debug 引入的新缺陷（最高优先验证项）

`fluid.ts:305-320` 的 billboard pass：

```ts
depthStencilAttachment: {
  view: this.sceneDepthView,
  depthLoadOp: 'load',    // 保留场景深度用于遮挡
  depthStoreOp: 'store',
}
```

pipeline 为 `depthWriteEnabled: true, depthCompare: 'greater'`（`fluid.ts:169-171`）。`depthLoadOp: 'load'` 实现了正确的遮挡剔除，但 `depthStoreOp: 'store'` + `depthWriteEnabled: true` 会把**通过测试的粒子深度写回 `sceneDepth`**。注释显示这是有意的（"Particle depths are written to sceneDepth so the surface shader later compares against the closest surface"）。

问题在于 surface pass 的遮挡判据（`surfaceShader.ts:90-93`）：

```wgsl
let sceneDepth = textureLoad(sceneDepthTexture, vec2<i32>(input.position.xy), 0);
if (sceneDepth > 0.0001 && fluidDepth < sceneDepth) { return sceneColor; }
```

此时 `sceneDepth` 在有粒子的像素上已经等于**该粒子未滤波的原始深度** `d_raw`，而 `fluidDepth` 是滤波后的 `d_filt`。该判据本意是"场景实体在流体前面就显示背景"，现在退化成了"`d_filt < d_raw` 就显示背景"。

由于双边滤波是邻域凸组合，且 `sigmaRange = 0.001`（NDC 单位）远大于相邻粒子的深度差，range 权重接近 1，中心值必然被邻域拉动：

- 球顶附近（局部最近点）邻域更远 → `d_filt < d_raw` → **判定为被遮挡，返回背景**；
- 球边缘（局部较远）邻域更近 → `d_filt > d_raw` → 显示流体。

**静态推断的画面**：每个粒子的中心区域被打洞透出背景，只在粒子交界处残留流体，整体呈蜂窝/网状孔洞。这比 C 原版"跑不起来"更容易被误判为"滤波强度不够"，因此需要优先验证。

最小验证方法见 8.1。最小修复方向是把 `depthStoreOp` 改为 `'discard'`（或另用一张只做遮挡测试的深度附件），使 surface pass 读到的仍是纯场景深度——这也正是 C 原版的做法，只是原版另配了一张 `billboardDepthStencil`，遮挡剔除没生效。

### 6.4 静态结论

C' 修复了原版全部会阻断启动的 API/WGSL 错误，具备了可运行性，是一次有效的 debug。但：任务 1（论文滤波）未完成；厚度缺失；颜色空间不一致；并且 #5 的修复方式引入了一个高风险的遮挡判据自毁问题。综合判定 **部分完成**。

## 7. 横向比较

### 7.1 论文语义

| 论文要素 | A Opus | B GPT | C' DeepSeek |
| --- | --- | --- | --- |
| Eq.2 过远样本钳制 | 有 | 有 | 无 |
| Eq.3/6 成对偏差修正 | 有 | 有 | 无 |
| Eq.8/9 动态深度范围 | 有 | 有 | 无 |
| Eq.5 深度相关核尺寸 | 有 | 有 | 无（固定 4 像素） |
| 背景/屏幕外边界语义 | 终止核 + 越界视为背景 | 背景混入加权 + 越界 clamp | 中心无流体直接返回 |
| 3.4 节 2D clean-up | 有（5×5） | 有（5×5） | 无 |
| 滤波所用深度空间 | 线性 eye z | 线性 eye z | 非线性 NDC |

```text
Opus > GPT >> DeepSeek(debug)
```

### 7.2 滤波核与 billboard 尺度（定量）

按 `fov = 50°`（`src/main.ts:66`）、相机 `(2,2,0)` 看原点（距离约 2.83）、模拟半径 `0.006`（`src/simulator/SPH.ts:43`）、`H ≈ 1800` 估算：

| 量 | A Opus | B GPT | C' DeepSeek |
| --- | --- | --- | --- |
| imposter 世界半径 | 0.012（2.0×） | 0.009（1.5×） | 0.006（1.0×） |
| 粒子屏幕投影半径 | ≈ 8.2 px | ≈ 6.1 px | ≈ 4.1 px |
| 滤波 σ | ≈ 5.7 px（未触 clamp 12） | ≈ 2.9 px（未触 clamp 6.67） | 固定 2.0 px |
| 滤波核半径 | ≈ 18 px | ≈ 9 px | 固定 4 px |
| 核半径 / 粒子半径 | ≈ 2.2 | ≈ 1.5 | ≈ 1.0 |

三点结论：

1. A 与 B 的 Eq.5 在默认参数下**都真正起作用**（σ 未被 clamp 饱和），深度相关核不是形同虚设；
2. A 的相对平滑量约为 B 的 1.5 倍，静态推断 A 的表面更连续、B 可能残留更多球状起伏；但 B 有 2 轮迭代 + clean-up 会累积平滑，最终差异需截图对比；
3. C' 的核半径与粒子投影半径相当（≈1.0），加之滤波不扩张流体区域，**结构上就无法消除粒子圆之间的空洞**——这一点与 6.3 的孔洞问题成因不同，需分别验证。

### 7.3 每帧开销（静态计数）

| 项 | 基线（球体） | A Opus | B GPT | C' DeepSeek |
| --- | --- | --- | --- | --- |
| 每粒子顶点数 | 81 顶点 / ≈96 三角形（`SphereGeometry(0.007,8,8)`，indexed 288 索引） | 4 × 2 pass = 8 | 6 × 2 pass = 12 | 6 × 1 pass = 6 |
| 粒子 draw call | 1 | 2 | 2 | 1 |
| render pass 数 | 1 | 1 + 3 = 4 | 1（+1 depth +1 thickness）= 3 | 5 |
| compute pass 数 | 0 | **1**（含 5 dispatch） | **5**（各 1 dispatch） | 0（滤波用 render pass） |
| 全屏 dispatch/draw | 0 | 5 dispatch + 1 draw | 5 dispatch + 1 draw | 2 draw + 1 draw |
| 屏幕尺寸中间纹理 | 1×depth32float | 2×r32float + 1×r16float | 3×r32float + 1×depth32float + 1×**rgba16float** | 3×r32float + 1×canvasFormat + 1×depth32float |
| 每像素字节（估） | — | 8 + 2 = **10 B** | 12 + 4 + 8 = **24 B** | 12 + 4 + 4 = **20 B** |

- 顶点量：三者相对基线都下降 1～2 个数量级，A 最省（8 vs 12）；
- 中间纹理：A 最省（B 的 `rgba16float` 厚度和第三张 r32float、C' 的整张 `sceneColor` 副本都是可省的）；
- pass 数：A 把 5 次 dispatch 合并进 1 个 compute pass；B 用了 5 个独立 pass；C' 用 render pass 做滤波并额外多一次全屏 `sceneColor` 拷贝语义；
- **B 有约一半滤波迭代是空转**（5.3.4），A 用 `break` 提前退出。

顶点量下降是结构证据，最终 GPU 时间仍必须 profiling 复核，且三者都**没有**为渲染 pass 添加 timestamp 埋点（基线的 `writeTimestamp(0)/(5)` 只覆盖首尾，中间 1–4 由 simulator 内部写入）。要满足 BENCHMARK.md 6.2 的分 pass 计时，需评测侧自行补埋点。

### 7.4 WebGPU API 质量

```text
Opus > GPT >> DeepSeek(debug) >>> DeepSeek(原始，无法创建管线)
```

- A：uniform 布局手算与 `OPTIONS_BUFFER_SIZE` 精确一致；r32float 只用 `textureLoad` + `unfilterable-float`；`depthReadOnly` 用法正确；compute pass usage scope 理解正确；`discard` 后 helper invocation 仍执行因此给 `sqrt` 加 `max` 保护（`common/shader.ts:103-105` 注释）。
- B：格式/usage/layout 全部匹配；ping-pong 链正确；但每帧 `createView()`、厚度格式过宽。
- C'：修复后类型匹配，但仍有重复 pipeline、无 view 缓存、遮挡语义自毁。

### 7.5 工程质量与可验证性

```text
Opus > GPT > DeepSeek(debug)
```

- A：改动聚焦且有抽象（`SphereImposter` 被两条路径共享）、参数全可调、保留原始粒子视图做对照、注释直接映射论文公式与 GPU 语义、无死代码；
- B：改动最小（`controller.ts` 仅 +6 行）、注释交代了参数来源，但参数硬编码、无调试视图、原始粒子渲染被直接替换；
- C'：无 UI、无调试视图、`rawParticles/` 死代码留存、有重复 pipeline。

### 7.6 静态可判定维度的评分

按 BENCHMARK.md 第 6 节维度。**视觉质量（20）与性能（15）无法静态判定，此处不给分**；其余 65 分给出静态评分：

| 维度 | 满分 | A Opus | B GPT | C' DeepSeek | C DeepSeek |
| --- | ---: | ---: | ---: | ---: | ---: |
| 功能完整性 | 30 | 29 | 26 | 14 | 12 |
| 图形与数学正确性 | 20 | 19 | 16 | 8 | 5 |
| WebGPU API 质量 | 10 | 10 | 8 | 6 | 1 |
| 工程质量 | 5 | 5 | 3 | 2 | 2 |
| **静态小计** | **65** | **63** | **53** | **30** | **20** |

扣分依据：

- A：alpha 混合在 gamma 空间（−1 图形数学）、折射不采 sceneColor（−1 功能）；
- B：无厚度预遮挡与场景预遮挡（−2 功能）、背景样本混入加权（−2 图形数学）、越界 clamp（−1 图形数学）、无厚度精度/格式优化与每帧 createView（−2 API）、参数硬编码无调试视图（−2 工程）、固定 20 次循环（−1 功能，性能相关部分留待 profiling）；
- C'：滤波非论文方法（−10 功能）、无厚度（−4 功能）、遮挡判据自毁（−6 图形数学）、颜色空间不一致（−3 图形数学）、法线 NaN 风险（−2 图形数学）、重复 pipeline 与无 UI（−4 API/工程）；
- C：在 C' 基础上追加 WGSL 编译与 bind group 创建失败（API 近乎归零，且硬性门槛 1/2/3 均不通过）。

## 8. 运行验收清单（按优先级）

静态结论要升级为最终结论，必须在同一设备、浏览器、分辨率、场景和相机下依次验证。以下按"能否推翻静态判断"排序。

### 8.1 最高优先：验证 C' 的孔洞推断（6.3）

最小方法：在 `deepseel_v4_pro_debug/src/renderer/fluid/fluid.ts:317` 把 `depthStoreOp: 'store'` 改为 `'discard'`，对比修改前后同一帧截图。

- 若修改后流体变连续，则 6.3 的推断成立，C' 的遮挡判据确实自毁；
- 若两者一致，则说明 `d_filt ≥ d_raw` 在实际参数下普遍成立，推断不成立，需修正本节结论。

对照检查：把 `fluidDepth < sceneDepth` 的判据临时改为常真/常假，观察流体覆盖范围变化。

### 8.2 高优先：硬性门槛

1. 四份代码分别 `yarn install && npm run build`（`tsc && vite build`，`strict: false`、`noUnusedLocals: false`，宽松，主要看类型错误）；
2. 启动页面，检查控制台 WGSL 编译错误、pipeline 创建错误、bind group 与 validation error；
   - C 预期在此处失败（缺 `DirectionalLight` 结构定义、`depth` sample type 不匹配）；
3. 确认 PBF 仿真与场景切换未回归（Bunny/Cube/Droplet/Dam Break/Boundary 全部可切换）。

### 8.3 中优先：定向验证各自的静态保留项

| 检查 | 目标 | 预期观察 |
| --- | --- | --- |
| Boundary + Water Droplet 场景 | B 的场景预遮挡缺失（5.3.3） | 物体轮廓附近是否有滤波深度/法线/厚度伪影 |
| 流体轮廓放大截图 | B 的背景样本混入（5.3.1） | 轮廓是否有一圈约一个粒子半径的深度下沉 |
| 屏幕四边流体 | B 的越界 clamp（5.3.2） | 边缘是否被额外拉伸平滑 |
| 半透明薄水花边缘 | A 的 gamma 空间混合（4.3.1） | 边缘是否偏亮 |
| 孤立单粒子/细水丝 | C' 的法线 NaN（6.2.4） | 是否出现异常亮点/黑点 |
| C' 与背景交界 | C' 颜色空间（6.2.3） | 流体是否明显偏暗、与背景不连续 |
| 低角度掠射 | A/B 的动态范围（Eq.8/9） | 连续表面是否被误判断层 |

### 8.4 性能

固定设备/浏览器/分辨率/粒子数/相机，记录平均 FPS 与 1% low、仿真暂停时的纯渲染 GPU 时间。要拿到分 pass 时间需自行补 timestamp 埋点（7.3）。重点验证两条静态推断：

- A 的 1 个 compute pass / 5 dispatch 是否优于 B 的 5 个 pass；
- B 的固定 20 次迭代空转是否可测量（对比把 `MAX_RADIUS` 调到 12 的版本）。

## 9. 错误标签

按 BENCHMARK.md 9.3 的标签体系：

| 版本 | 标签 |
| --- | --- |
| A Opus | （无阻断级标签）`color_space_minor`（gamma 空间 alpha 混合） |
| B GPT | `filter_boundary_artifact`（背景样本混入 + 越界 clamp）、`insufficient_scene_occlusion`（深度/厚度 pass 未用场景深度）、`resource_waste`（rgba16float 厚度、每帧 createView、固定 20 次迭代）、`insufficient_verification`（无调试视图与参数 UI） |
| C' DeepSeek (debug) | `not_narrow_range_filter`、`missing_thickness`、`depth_space_error`（在 NDC 深度上滤波）、`occlusion_logic_error`（6.3，新增标签）、`color_space_error`、`normal_reconstruction_error`（NaN 风险）、`resource_waste`（重复 pipeline）、`insufficient_verification` |
| C DeepSeek (原始) | C' 全部标签 + `wgsl_compile_error`（缺 `DirectionalLight`）、`webgpu_validation_error`（sample type / uniform 尺寸）、`billboard_occlusion_error`、`uv_flip_error` |

## 10. 给评测流程的两点建议

1. **C 与 C' 应作为两条独立记录归档**。C' 包含人工介入，按 BENCHMARK.md 第 3 节的协议，追加提示与人工修复必须逐条记录。6.1 的表格可直接作为 C' 的 `metadata.md` 中"人工介入内容"一栏。以 C 参与四模型排序、以 C' 说明"经人工 debug 后的上限"，比只保留一份更有信息量。
2. **本任务的区分度主要来自边界条件，而不是主干管线**。三份代码都能搭出"billboard → 深度 → 滤波 → 合成"的主干；真正拉开差距的是：背景/屏幕外样本怎么处理、厚度累加与深度剔除的先后、uniform 布局手算、compute pass 的 usage scope、法线退化保护、颜色空间一致性。如果后续要提高评测难度，应当围绕这些点设计检查项（例如强制非 32 倍数分辨率、要求多层深度相邻的流体、加入需要正确 alpha 合成的半透明背景），而不是继续增加"是否实现了某个 pass"这类可以被模仿满足的条件。

---

## 11. R — 参考实现 `main`（`49a38e3`）的同标准评审

第 1–10 节的评审在不参考 `main` 的前提下完成。本节把同一套标准施加到参考实现上，目的不是给作者打分，而是**校准评分刻度**：确认"论文完整度"和"工程质量"这两条尺子上，参考实现分别处在什么位置。

`main` 有两套流体渲染器，本节只评审带滤波的 `src/renderer/filteredParticleFluid/`（10 个文件，约 1250 行）。另一套 `src/renderer/particleFluid/` 是无滤波版本，不在对比范围内。

### 11.1 管线

```text
主 pass（mesh + skybox → canvas, renderDepthMap）
  → depth pass    4 顶点 strip imposter → r32float(正值 eye 距离)，以 renderDepthMap 为深度附件
  → volume pass   4 顶点 strip imposter（quad 扩大 2×）加法混合 → r16float，半分辨率，无深度附件
  → filter        1 个 compute pass：X 轴 dispatch + Y 轴 dispatch，各 1 轮
                  （workgroup shared memory，workgroup_size(32,1,1)，shared array<f32,64>）
  → render pass   4 顶点全屏 quad，renderBundle，src-alpha 混合覆盖到 canvas
```

深度约定与三个模型都不同：`depthCam = -positionCam.z`，即**正值** eye 距离，"值更小 = 更近"。A/B 用负值 eye z（"值更大 = 更近"），C' 用 Reverse-Z NDC。三种约定都自洽，`main` 与 A/B 同属线性 eye 深度。

### 11.2 参考实现独有的工程优势（三个模型均未做到）

1. **workgroup shared memory 滤波**（`shader/filterPassShader.ts:31-46`）。`workgroup_size(32,1,1)` + `array<f32, 64>` 共享缓冲，每线程只做 2 次 `textureLoad` 预载左右半区，`workgroupBarrier()` 后全部从 shared memory 取样。索引映射验算：`sharedBuffer[j] ↔ tex(base + j − 16)`，覆盖 `tex(base−16)` 到 `tex(base+47)`，恰好满足 32 个线程各访问 `center ± 16` 的需求，边界零浪费。
   - 对比：A 每像素最多 2×18 = 36 次 `textureLoad`，B 固定 2×20 = 40 次。`main` 是 2 次。
   - 这是**本次对比中最显著的性能设计差异**，且是三个模型都没有想到的方向。
2. **厚度图半分辨率**（`fluid.ts:35`，`size: [width/2, height/2]` + `r16float`）。厚度是低频信号，半分辨率足够，填充率与带宽降到 1/4。A 用全分辨率 `r16float`，B 用全分辨率 `rgba16float`（是 `main` 的 8 倍）。
3. **7 种调试显示模式**（`shader/renderPassShader.ts:118-146`：PBR / PBR无折射 / Diffuse / Normal / Depth / Thickness / Position），可直接肉眼验证每个中间缓冲。A 只有"流体表面 / 原始粒子"切换加参数，B 和 C' 完全没有。这是**可验证性上的代差**。
4. **GPU timestamp 分 pass 埋点**（`fluid.ts:150-172`，timestamp 6/7/8 分别切分光栅化、滤波、合成）。三个模型都没有加，导致 BENCHMARK.md 6.2 要求的分 pass GPU 时间无法直接采集。
5. **Fresnel F0 由折射率推导**（`renderPassShader.ts:73-75`：`F_0 = ((n−1)/(n+1))²`）而非硬编码，并用 UE4 的球面高斯近似 `exp2((−5.55473·VoH − 6.98316)·VoH)` 替代 `pow(1−x,5)`。A 硬编码 0.02、B 硬编码 0.02037，数值等价但不如从 IOR 推导严谨。
6. **`filterSize < 2` 时整体关闭滤波**（`fluid.ts:120`），可在运行时直接对照"有/无滤波"。
7. **`dpdx/dpdy` 求法线，且 `discard` 刻意放在导数计算之后**（`renderPassShader.ts:150-153`）。这个顺序是必需的——若先 discard，2×2 quad 内的导数会失效。与 A 在 `common/shader.ts:103-105` 处理 helper invocation 的注释属于同一类 GPU 语义知识。
8. **合成 pass 用 `writeMask: RGB`**（`screenSpaceRenderer.ts:69`）不写 alpha 通道；volume pass 用 `writeMask: RED`（`paricleRasterizer.ts:127`）。细节干净。

### 11.3 参考实现的缺陷（按严重度）

**11.3.1 shared memory 中心索引错位——`filterSize ≠ 32` 时滤波结果整体平移（真实 bug）**

`shader/filterPassShader.ts` 中：

```wgsl
const HalfMaxFilterSize: u32 = MaxFilterSize >> 1;   // 恒为 16
let HalfFilterSize: u32 = options.filterSize >> 1;   // 可变

// 预载按 HalfMaxFilterSize(16) 偏移
sharedBuffer[thread_id] = textureLoad(srcTexture, texCoord - i32(HalfMaxFilterSize) * filterDirection, 0).r;
...
// 取中心却按 HalfFilterSize 偏移
let sharedBufferCenterIndex = thread_id + HalfFilterSize;
```

预载建立的映射是 `sharedBuffer[j] ↔ tex(base + j − 16)`，因此线程 `t` 的中心必须取 `sharedBuffer[t + 16]`。代码取的是 `sharedBuffer[t + filterSize/2]`：

- `filterSize = 32`（**默认值**）→ `HalfFilterSize = 16` → 正确；
- `filterSize = 16` → 取 `sharedBuffer[t+8]` ↔ `tex(base+t−8)`，却写入 `tex(base+t)` → 沿滤波方向**偏移 8 像素**；
- `filterSize = 2` → 偏移 15 像素。

X 与 Y 两轮各偏移一次，流体表面会整体平移 `(16 − filterSize/2, 16 − filterSize/2)` 像素，与厚度图和场景错位。UI 允许 `filterSize ∈ {0,2,...,32}`（`config.ts` 的 `initRenderingOptions`），因此只要用户调小滤波尺寸就会触发。默认值恰好落在唯一正确的取值上，所以该 bug 在默认配置下不可见。

最小验证：把 `filterSize` 从 32 调到 16，观察流体是否相对场景整体平移约 8 像素。

**11.3.2 quad 半宽与重建球半径相差 2 倍**

`shader/depthPassShader.ts`：vertex 用 `positions ∈ ±0.5` 乘 `options.radius`，quad 半宽 = `0.5·radius`；fragment 用 `normalCam.xy = uv·2−1 ∈ ±1` 乘 `options.radius`，重建球面 xy 偏移最大 = `radius`。

即屏幕上的圆对应世界半径 `0.5·radius`，而写入的球面深度按半径 `radius` 计算，**沿视线方向的深度凸起被夸大 2 倍**（默认 `radius = 0.02` 时，屏幕圆半径 0.01，中心深度凸起 0.02）。每个粒子实际被渲染成沿视线拉长 2 倍的椭球。滤波会平掉大部分起伏，但边缘曲率会系统性偏强。

A 与 B 在这一点上都是自洽的（quad 半宽 = 球半径 = r），**数学严格性上优于参考实现**。

**11.3.3 volume pass 无场景深度剔除**

`paricleRasterizer.ts:191-200`：volume pass 只有 colorAttachment，没有 depthStencilAttachment。被实体遮挡的粒子仍累加厚度。与 B 的 5.3.3 是同一个问题。A 是四者中唯一用 `depthReadOnly: true` 正确处理的（既剔除被遮挡粒子，又允许表面后的粒子累加）。

**11.3.4 `dpdx/dpdy` 法线无断层保护**

`renderPassShader.ts:39-44` 直接对重建的 eye position 求硬件导数。优点是代码极简、无需邻域采样；缺点是：

- 在深度断层处导数跨越断层，法线被严重拉扯，没有任何判据剔除；
- 前向差分精度低于中心差分，且以 2×2 quad 为单位，边界可能有块状 artifact。

A 有"两侧都有流体时取深度变化较小的一侧"，B 有 `abs(z − center) < 0.12` 阈值。**这正是 BENCHMARK.md 6.1 中 Water Droplet 场景要检查的"前景液滴与底层液面之间的深度断层是否被错误连接"，参考实现在此项上弱于 A 和 B。**

**11.3.5 论文语义缺两项**

- **无 Eq.5 深度相关核尺寸**：`Sigma = f32(HalfFilterSize) * 0.5` 是固定像素核，不随眼空间深度变化。BENCHMARK.md 10.3 已声明这是工程简化。
- **无 3.4 节的 2D clean-up pass**，且只做 1 轮 H+V（`fluid.ts` 只有一张 `fluidDepthMap` 加一张 `tempTexture`，结构上无法多轮迭代）。

**11.3.6 厚度不是物理弦长**

`volumePassShader.ts`：`volume = exp(-radius2 * 2.0)`，是无量纲高斯核而非弦长 `2r√(1−d²)`，因此 `exp(−opacity·thickness)` 中的 `thickness` 没有长度单位，需要靠合成端的魔数 `0.02`（`renderPassShader.ts:155`）和 `opacity = 0.25` 补偿。A 与 B 都用了物理弦长，Beer-Lambert 的量纲是对的。

**11.3.7 主模式不透明**

`mode = 0`（默认 PBR）输出 `alpha = 1.0`，合成 pass 虽然配了 src-alpha 混合，实际是完全覆盖；只有 `mode = 1`（无折射）才有 `alpha = 1 − attenuate`。薄水花与飞溅在默认模式下不会透出背景。A 是四者中唯一在主路径上做 alpha 混合的（`alpha = 1 − exp(−opacity·thickness)`）。

**11.3.8 次级**

- `paricleRasterizer.ts:36` 的 `initRenderBundle()` 被注释掉，两个 bundle 字段成为死代码（合成 pass 的 bundle 有在用）；
- `textureFilter.ts:130-140` 用一个空 render pass 来 clear storage texture——WebGPU 没有 clearTexture API，这是合法且必要的技巧，但每帧多一个 pass；
- 每帧 `createView()`（`textureFilter.ts:132`、`paricleRasterizer.ts:178`、`191`），与 B 相同；
- `filterPassShader.ts` 末尾保留了 20 行被注释的旧双边滤波代码。

### 11.4 参考实现的静态评分（同一把尺子）

| 维度 | 满分 | R main | A Opus | B GPT | C' DeepSeek |
| --- | ---: | ---: | ---: | ---: | ---: |
| 功能完整性 | 30 | 25 | 29 | 26 | 14 |
| 图形与数学正确性 | 20 | 14 | 19 | 16 | 8 |
| WebGPU API 质量 | 10 | 9 | 10 | 8 | 6 |
| 工程质量 | 5 | 4 | 5 | 3 | 2 |
| **静态小计** | **65** | **52** | **63** | **53** | **30** |

R 的扣分：无 Eq.5、无 clean-up、只 1 轮迭代（−3 功能）、厚度非物理弦长（−1 功能）、volume 无深度剔除（−1 功能）、quad/球半径 2 倍不一致（−2 图形数学）、shared memory 索引错位（−2 图形数学）、dpdx 法线无断层保护（−2 图形数学）、主模式不透明（−1 图形数学）、每帧 createView 与空 pass clear（−1 API）、死代码与注释残留（−1 工程）。

R 的加分项（shared memory、半分辨率厚度、timestamp 埋点、7 种调试模式、F0 由 IOR 推导）主要落在**性能与视觉质量**这两个静态不给分的维度上，因此 52 分低估了它的实际工程水平。**这正是本表的用法边界：静态 65 分只衡量"论文语义 + 数学正确性 + API 规范 + 代码组织"，不衡量"跑得多快、看起来多好"。**

## 12. 四方对比与对评测流程的影响

### 12.1 论文语义完整度

| 论文要素 | R main | A Opus | B GPT | C' DeepSeek |
| --- | --- | --- | --- | --- |
| Eq.2 过远样本钳制 | 有 | 有 | 有 | 无 |
| Eq.3/6 成对偏差修正 | 有 | 有 | 有 | 无 |
| Eq.8/9 动态深度范围 | 有 | 有 | 有 | 无 |
| Eq.5 深度相关核尺寸 | **无**（固定像素核） | 有 | 有 | 无 |
| 3.4 节 2D clean-up | **无** | 有 | 有 | 无 |
| 迭代轮数 | 1 轮 H+V | 可调，默认 2 轮 | 固定 2 轮 | 1 轮 |
| 滤波所用深度空间 | 线性 eye（正值） | 线性 eye（负值） | 线性 eye（负值） | 非线性 NDC |

```text
论文完整度：Opus ≈ GPT > main >> DeepSeek(debug)
```

**A 与 B 在论文完整度上超过了参考实现**（多出 Eq.5 与 clean-up）。这与 BENCHMARK.md 第 99 行的预设一致：深度相关核尺寸应当加分，而不是因为偏离参考代码而扣分。本次结果验证了那条评分规则确有必要——若以 `main` 为基准做相似度比较，会把 A 和 B 做对的部分判成偏离。

### 12.2 工程与性能设计

| 项 | R main | A Opus | B GPT | C' DeepSeek |
| --- | --- | --- | --- | --- |
| 每粒子顶点数 | 4 × 2 pass = 8 | 4 × 2 pass = 8 | 6 × 2 pass = 12 | 6 × 1 pass = 6 |
| 滤波每像素 textureLoad | **2**（shared memory） | ≤ 36 | 40（固定） | 18 |
| 厚度分辨率/格式 | **半分辨率 r16float** | 全分辨率 r16float | 全分辨率 rgba16float | 无厚度 |
| 场景深度剔除粒子 | 深度 pass 有，厚度 pass 无 | **两个 pass 都有** | 两个都无 | 深度 pass 有（但见 6.3） |
| 法线断层保护 | **无**（dpdx/dpdy） | 有（取深度变化小的一侧） | 有（阈值 0.12） | 无（且有 NaN 风险） |
| 中间缓冲调试视图 | **7 种** | 部分（表面/粒子切换） | 无 | 无 |
| 分 pass timestamp | **有** | 无 | 无 | 无 |
| 主路径 alpha 混合 | 无（mode 0 不透明） | **有** | 无 | 无 |
| 颜色空间一致 | 是 | 是 | 是（1/2.2 近似） | **否** |

```text
性能设计：main > Opus > GPT >> DeepSeek(debug)
可验证性：main > Opus >> GPT ≈ DeepSeek(debug)
数学严格性与边界处理：Opus > GPT > main >> DeepSeek(debug)
```

三条尺子上的排序不一致，这本身是有价值的结论：**参考实现赢在"跑得快、看得见"，两个强模型赢在"论文更完整、边界更严谨"。**

### 12.3 三个可迁移的优化点

以下是参考实现做到、而没有任何模型想到的方向，可以作为后续加难度的检查项：

1. **workgroup shared memory 做可分离滤波**（2 次 load vs 36–40 次）。这是本任务最大的性能杠杆，没有一个模型使用。若在 Prompt 中加入明确的带宽/GPU 时间预算，应能把这个方向逼出来。
2. **厚度图降分辨率**。低频信号降采样是常规手段，三个模型都用了全分辨率。
3. **自带分 pass timestamp 与中间缓冲可视化**。三个模型都没有主动为自己的实现建立可观测性——而 BENCHMARK.md 6.2 明确要求分 pass GPU 时间。这直接对应 `insufficient_verification` 标签：模型交付了功能，但没有交付验证手段。

### 12.4 对评测流程的三点结论

1. **`main` 不能作为满分答案，也不适合做相似度基准。** 它在论文完整度上缺 Eq.5 与 clean-up（BENCHMARK.md 10.3 已说明），在数学严格性上有 11.3.1、11.3.2 两处实际问题。按同一套静态标准它得 52 分，低于 Opus 的 63 分。BENCHMARK.md 10.3 的表述可以进一步收紧为："`main` 是性能与可验证性的参考上界，不是论文实现的参考答案。"
2. **建议把"是否自带可观测性"提升为独立评分项。** 目前 BENCHMARK.md 的 6 个维度里，调试视图和 timestamp 埋点只能挤进"工程质量（5 分）"，权重与它的实际价值不匹配。参考实现与三个模型在这一项上的差距（7 种模式 + 3 个埋点 vs 0）比在功能完整性上的差距更大。
3. **建议把 11.3.1 作为"缺陷参考实现"样例。** BENCHMARK.md 第 246 行提到"加入缺陷参考实现，让模型识别并修复"。`filterSize ≠ 32` 时的 shared memory 索引错位是一个理想素材：默认配置下完全不可见、需要理解 shared memory 索引映射才能定位、修复只需一处改动（把 `HalfFilterSize` 换成 `HalfMaxFilterSize`），且能明确区分"读懂了代码"和"看起来读懂了"。
