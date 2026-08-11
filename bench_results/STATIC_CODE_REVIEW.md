# Code Agent 流体渲染结果静态 Review

## 1. Review 范围

本文档对以下三个 Code Agent 结果进行静态代码评审：

| 模型 | 结果目录 |
| --- | --- |
| GPT 5.6 Sol Medium | `bench_results/gpt_5_6_sol_medium` |
| Opus 5 | `bench_results/opus_5` |
| DeepSeek V4 Pro | `bench_results/deepseek_v4_pro` |

评审基准包括：

- 测试起点 `benchmark/raw-particles-baseline`；
- 项目作者在 `main` 中实现的参考管线；
- `NarrowRangeFilter.pdf` 描述的方法；
- Benchmark Prompt 中提出的三项任务。

本轮只阅读代码，没有安装依赖、构建项目、编译 WGSL、启动浏览器或观察实际画面。因此，“成功”表示代码在功能设计上覆盖了任务要求，不等价于已经通过运行验收。WebGPU validation、画面质量和性能结论仍需在统一环境中复核。

## 2. 任务拆解

需要检查的三个核心目标为：

1. 对粒子深度图进行论文所述的 Narrow-Range 平滑；
2. 在 screen space 重建并渲染流体表面，包含环境反射和折射；
3. 用 billboard 取代原有的低模球体实例渲染。

辅助检查项包括：

- 是否生成厚度或体积近似；
- 深度空间、Reverse-Z 和坐标变换是否正确；
- 流体与场景网格之间是否正确遮挡；
- WebGPU texture format、sample type、usage 和 WGSL 声明是否匹配；
- 是否存在会导致 pipeline 创建失败或画面明显错误的问题；
- 实现是否保留原有 PBF 仿真和场景切换。

## 3. 总体结论

| 模型 | 深度滤波 | Screen-space 流体 | Billboard | 静态完成状态 |
| --- | --- | --- | --- | --- |
| Opus 5 | 完成，较完整复现论文 | 完成 | 完成，4 顶点 | **成功，当前最佳** |
| GPT 5.6 Sol Medium | 完成，较完整复现论文 | 完成 | 完成，6 顶点 | **成功** |
| DeepSeek V4 Pro | 未完成论文算法，当前为双边滤波 | 基本完成，但缺少厚度 | 完成，6 顶点 | **部分完成，存在运行风险** |

静态排序：

```text
Opus 5 > GPT 5.6 Sol Medium >> DeepSeek V4 Pro
```

Opus 和 GPT 都形成了完整的“粒子 billboard -> 深度/厚度 -> 滤波 -> 法线重建 -> 环境反射折射 -> 最终合成”链路。Opus 在场景遮挡、参数化和每粒子顶点数方面更好。DeepSeek 搭出了多 pass 框架，但当前滤波并不是 Narrow-Range Filter，且代码快照中仍有若干资源布局和参数传递问题。

## 4. Opus 5

### 4.1 已完成的功能

- 使用每粒子 4 顶点的 `triangle-strip` billboard；
- 在 fragment shader 中丢弃圆外像素；
- 近似恢复球面前表面的眼空间深度，并写入 Reverse-Z 深度；
- 使用 `r16float` 纹理和加法混合生成厚度图；
- 根据眼空间深度、屏幕高度与相机 FOV 动态计算屏幕空间滤波核；
- 实现范围外深度钳制、对称样本偏差修正和动态范围调整；
- 支持多轮横向/纵向 1D pass；
- 实现论文建议的 5x5 二维 clean-up pass；
- 从滤波深度重建眼空间位置与法线；
- 在统一的世界空间中计算反射和折射方向；
- 查询 cubemap 获得环境反射与折射；
- 使用 Schlick Fresnel、Beer-Lambert 吸收和厚度透明度；
- 深度与厚度 pass 都使用场景深度剔除被实体遮挡的粒子；
- 提供流体表面/原始粒子切换及完整的渲染参数 UI。

### 4.2 相对参考实现的特点

Opus 并没有逐行复制 `main`。它在参考管线基础上进一步实现了：

- 深度相关的投影核尺寸；
- 多轮可分离滤波；
- 最终二维 clean-up；
- 更完整的厚度吸收和透明混合；
- 更系统的参数配置。

这些设计比 `main` 中固定像素核、两次 1D pass 的实现更接近论文的完整描述。

### 4.3 静态保留项

1. Billboard quad 使用切锥半径进行扩大，而 fragment 中仍采用正交球面投影近似。粒子靠近相机时，光栅位置和写入的近似球面深度之间会存在误差。题目明确允许近似，因此不影响完成状态。
2. 默认执行两轮 H/V 滤波并附加 clean-up，最大核半径达到 48 像素，可能带来较高的全屏 GPU 成本，需要 profiling 后评价。
3. 最终颜色带透明度并在普通 UNORM canvas 上混合；如果 shader 已输出 gamma 编码颜色，固定功能混合发生在非线性颜色空间，可能影响半透明边缘，但不阻断核心功能。

### 4.4 静态结论

三项任务全部完成。代码体现了对论文、WebGPU 资源绑定、Reverse-Z、坐标空间和屏幕空间流体渲染的较深入理解。

## 5. GPT 5.6 Sol Medium

### 5.1 已完成的功能

- 使用每粒子 6 顶点、两个三角形的 billboard；
- 根据局部圆坐标计算球面前表面深度；
- 独立生成眼空间深度、设备深度和厚度纹理；
- 使用加法混合累计粒子弦长作为流体厚度；
- 根据世界空间 sigma、眼空间深度和投影矩阵计算屏幕空间核；
- 实现对称样本、深度钳制和动态范围调整；
- 执行两轮横纵 Narrow-Range Filter；
- 添加 5x5 clean-up pass；
- 从滤波深度重建眼空间位置和法线；
- 正确转换到世界空间查询 cubemap；
- 使用约 `1.333` 的水折射率、Schlick Fresnel 和厚度吸收；
- 最终合成写入 `frag_depth`，可参与主场景 Reverse-Z 深度测试。

### 5.2 主要问题

#### 场景遮挡发生得太晚

粒子深度 pass 使用私有的 `particleDepthTexture`，厚度 pass 没有场景深度附件。实体后面的粒子因此仍会进入深度滤波和厚度累积。

最终 composite pass 写入流体深度，可以让场景实体遮挡最终表面；但在此之前，隐藏粒子已经可能影响：

- 物体轮廓附近的滤波深度；
- 由深度差分得到的法线；
- 流体厚度与吸收。

这可能在流体与圆环等实体相交时产生轮廓伪影。Opus 在粒子预处理阶段就使用场景深度，处理更完整。

#### 工程可调性较弱

- 粒子半径、滤波参数、吸收系数等基本硬编码；
- 没有原始深度、滤波深度、法线和厚度等调试显示模式；
- 固定执行四次 1D pass 和一次 clean-up，缺少运行时性能/质量取舍。

#### 顶点数量不是最精简方案

每粒子使用 6 顶点，而 Opus 和 `main` 使用 4 顶点 `triangle-strip`。相比原始低模球体仍然是显著优化，只是在三个候选中不是最优。

### 5.3 静态结论

三项任务全部完成。其论文滤波和最终着色实现较扎实，主要差距集中在场景预遮挡与工程可调性，而不是核心功能缺失。

## 6. DeepSeek V4 Pro

### 6.1 已完成的功能

- 使用每粒子 6 顶点 billboard 取代球体网格；
- 丢弃投影圆外像素；
- 近似计算球面前表面的 Reverse-Z 深度；
- 建立场景颜色、场景深度、粒子深度、横向滤波、纵向滤波和最终合成 pass；
- 从滤波深度重建位置和法线；
- 使用 cubemap 查询环境反射和折射；
- 加入 Fresnel 与方向光高光；
- 最终按场景深度判断流体是否被实体遮挡。

### 6.2 滤波算法评估

当前 shader 使用：

```wgsl
spatialW = exp(-offset * offset / (2 * sigmaSpatial * sigmaSpatial));
rangeW = exp(-depthDiff * depthDiff / (2 * sigmaRange * sigmaRange));
weight = spatialW * rangeW;
```

这是标准可分离双边滤波，公式本身成立，但没有实现论文 Narrow-Range Filter 的关键行为：

- 范围外远景深度的钳制；
- 中心两侧样本成对处理的偏差修正；
- 沿连续表面动态扩展允许深度范围；
- 论文定义的深度相关屏幕核尺寸。

因此任务 1 不能判为完成。本文只记录该差异，不要求在修复运行问题时同时更换算法。

### 6.3 为什么双边滤波后仍有明显小球

当前默认值为：

```text
particleRadius = 0.006
kernelRadius = 4
sigmaSpatial = 2
sigmaRange = 0.001（Reverse-Z/NDC 深度）
H/V 轮数 = 1
```

存在四个直接原因：

1. 半径 4 的滤波核每个方向只有 9 个采样，明显小于 `main` 默认约 16 像素的半径；
2. billboard 渲染半径与模拟半径相同，粒子投影之间近似相切，没有为表面重建提供足够重叠；
3. 无流体中心像素直接返回 0，滤波不会扩张有效流体区域，无法填补圆之间的空洞；
4. 在非线性的 NDC 深度上使用固定 `sigmaRange`，近处容易过度保留单球深度，远处的行为又不同。

仅从参数上改善双边滤波，可以先验证以下范围：

```text
particleRadius: 0.010～0.012
kernelRadius: 12～16
sigmaSpatial: 4～8
sigmaRange: 0.002～0.005（仅适用于暂时继续使用当前 NDC 深度时）
H/V 轮数: 2～3
```

其中最优先的不是盲目扩大核，而是适当扩大渲染 billboard 半径。因为当前滤波不会给背景像素生成新深度，只增大核仍无法消除粒子圆之间的空洞。

### 6.4 当前代码快照中的实现问题

以下结论以目前 `bench_results/deepseek_v4_pro` 中可见的文件为准。若模型已经在其他位置修复，需要将最新版本同步到结果目录后重新 review。

#### 1. 深度纹理采样的 level 参数类型错误

`texture_depth_2d` 的 `textureSampleLevel` 重载要求 level 为 `i32` 或 `u32`，不能使用 `0.0`。应使用整数 level，或直接按像素使用 `textureLoad`。

#### 2. 场景深度 bind group 类型不匹配

场景深度为 `depth32float`，WGSL 声明为 `texture_depth_2d`，但 layout 中声明为 `unfilterable-float`。正确的 sample type 应为 `depth`。

#### 3. `FilterParams` uniform buffer 尺寸不足

WGSL 结构包含一个 `vec2<f32>` 和三个标量，对齐后的结构尺寸为 24 字节。CPU 端当前只有 5 个 float，即 20 字节，需要补足 padding。

#### 4. H/V 方向共享同一个被覆盖的 uniform buffer

横向和纵向 pass 在 command buffer 提交前分别把同一个 buffer 写成 1 和 0。两个 pass 可能都看到最终值 0，实际执行两次纵向滤波。应使用独立 buffer、动态 offset 或 pipeline constant。

#### 5. Billboard 没有使用场景深度

粒子使用独立的深度附件，隐藏在实体后的粒子仍然进入滤波。最终深度比较只能阻止最终着色覆盖实体，不能撤销隐藏粒子对滤波法线的影响。

#### 6. 法线朝向判断存在反向风险

当前使用 `normalize(viewPos)` 作为 view direction，实际它指向从相机到表面的方向。后续翻转条件会将面向相机的法线翻到背面，导致环境反射和折射方向错误。应使用指向相机的 `normalize(-viewPos)` 进行判断，并同时确认 `cross` 顺序与屏幕 Y 方向。

#### 7. 全屏三角形 UV 可能垂直翻转

当前 position/UV 映射采用 OpenGL 风格的 Y 方向，而 WebGPU framebuffer 与纹理坐标以左上为原点。场景颜色和深度后处理可能整体上下翻转。

#### 8. 无粒子 fallback pass 缺少深度附件

当关闭粒子或粒子数为 0 时，fallback pass 没有 depth attachment，却调用带深度状态的 skybox/mesh pipeline，可能触发 render pass 与 pipeline 不兼容的 validation error。

#### 9. 缺少厚度或体积近似

当前没有厚度 pass。折射和反射可以工作，但透射颜色不会随流体厚度变化，薄水层、厚水体和粒子飞溅缺乏可区分的吸收效果。

#### 10. 输出颜色空间可能不一致

环境贴图通过 sRGB view 读取后得到线性颜色，最终直接写入普通 UNORM canvas，没有执行与项目其他渲染路径一致的 gamma 编码，画面可能明显偏暗。

### 6.5 静态结论

DeepSeek 完成了 billboard 和环境反射/折射的主要框架，但没有完成论文滤波任务。当前代码快照还包含可能阻止 pipeline 创建或导致错误画面的实现问题，因此只能判定为部分完成。

## 7. 横向比较

### 7.1 论文理解

```text
Opus ≈ GPT >> DeepSeek
```

Opus 和 GPT 都实现了论文区别于普通双边滤波的核心语义。DeepSeek 使用的是常规 bilateral Gaussian。

### 7.2 WebGPU 管线完整性

```text
Opus > GPT >> DeepSeek
```

Opus 在粒子预处理阶段就与场景深度结合。GPT 的最终深度测试较正确，但隐藏粒子仍可能污染中间缓冲。DeepSeek 当前存在明确的资源类型和 uniform 对齐问题。

### 7.3 Billboard 效率

```text
Opus（4 顶点）> GPT ≈ DeepSeek（6 顶点）>> 基线球体网格
```

三者都显著优于基线；Opus 与 `main` 的 4 顶点方案最精简。

### 7.4 最终水体表现的代码上限

```text
Opus > GPT >> DeepSeek
```

Opus 和 GPT 都有厚度、吸收、Fresnel、反射和折射。DeepSeek 只有表面环境着色，没有厚度驱动的体积感。

### 7.5 工程可验证性

```text
Opus > GPT > DeepSeek
```

Opus 提供原始粒子切换和丰富参数，便于定位问题。GPT 功能完整但硬编码较多。DeepSeek 缺少中间缓冲调试显示和渲染参数 UI。

## 8. 运行验收建议

静态 review 完成后，应在相同设备、浏览器、分辨率和相机下依次检查：

1. TypeScript 与 Vite 构建；
2. WGSL 编译和 pipeline 创建；
3. 浏览器控制台是否存在 WebGPU validation error；
4. 原始粒子深度、滤波深度、法线和厚度中间结果；
5. Water Droplet 场景中的前后景深度断层；
6. Boundary 场景中的流体/实体遮挡；
7. 低角度观察下的表面连续性；
8. Double Dam Break 中的时序稳定性和 GPU 时间；
9. 固定相机截图与视频；
10. 与基线及 `main` 的同条件对比。

只有通过以上检查后，才能把静态“成功”升级为最终“成功”，并据真实性能和视觉结果确定最终模型排序。
