# 无头 Linux 测试平台的 Vulkan / wgpu-py 配置

本文记录在测试平台容器中启用软件 Vulkan，并让 `wgpu-py` 在没有浏览器、
显示器和物理 GPU 的 Linux 环境里运行 compute shader 的方法。

已验证的环境特征：

- Debian 13 / Python 3.12；
- Vulkan loader 1.4；
- Mesa 25.0.7；
- `llvmpipe` CPU 设备；
- `wgpu-py 0.32.0`，Vulkan backend。

## 1. 确认容器身份和系统发行版

测试容器的 shell 已经是 `root`，所以不需要、通常也没有 `sudo`：

```bash
id
cat /etc/os-release
```

以下命令直接使用 `apt-get`。出现 `sudo: command not found` 不表示安装失败，
只需去掉 `sudo`。

## 2. 刷新软件包索引

最初执行安装时出现过下面的错误：

```text
Error: Unable to locate package mesa-vulkan-drivers
Error: Unable to locate package libvulkan1
```

这是 APT 包索引尚未下载或 Debian 软件源未配置造成的，不是包名错误。先运行：

```bash
apt-get update
```

如果 `apt-get update` 没有访问任何 Debian 仓库，先检查镜像内的软件源：

```bash
find /etc/apt -maxdepth 2 -type f -print
cat /etc/apt/sources.list 2>/dev/null || true
cat /etc/apt/sources.list.d/debian.sources 2>/dev/null || true
```

如果使用的是 Debian 13（`VERSION_CODENAME=trixie`），且镜像中确实没有软件
源，可以创建 `/etc/apt/sources.list.d/debian.sources`：

```text
Types: deb
URIs: https://deb.debian.org/debian
Suites: trixie trixie-updates
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: https://security.debian.org/debian-security
Suites: trixie-security
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
```

测试平台如果提供内部 APT 镜像，应把上述 URI 换成平台提供的镜像地址。写入后
重新执行：

```bash
apt-get update
```

不要在未确认 `/etc/os-release` 的情况下照搬 `trixie` 源到其他 Debian 或
Ubuntu 版本。

## 3. 安装最小无头 Vulkan 运行时

```bash
apt-get install -y \
    libvulkan1 \
    mesa-vulkan-drivers \
    vulkan-tools
```

各软件包的用途：

- `libvulkan1`：Vulkan loader；
- `mesa-vulkan-drivers`：Mesa 软件 Vulkan 驱动，提供 llvmpipe/Lavapipe；
- `vulkan-tools`：提供 `vulkaninfo`，用于诊断和验收。

这里只执行离屏 compute pass，不需要安装 `xserver-xorg-core`、窗口管理器、
浏览器或 `libegl1-mesa-dev`。它们不是运行本测试的前置条件。

## 4. 配置无头运行时目录

容器内没有桌面会话时，`XDG_RUNTIME_DIR` 往往未设置。为当前 shell 创建一个：

```bash
install -d -m 700 /tmp/wgpu-runtime-root
export XDG_RUNTIME_DIR=/tmp/wgpu-runtime-root
unset DISPLAY
unset WAYLAND_DISPLAY
```

这不会启动显示服务器；它只是给 Vulkan/Mesa 提供一个权限正确的运行时目录。

## 5. 验证 Vulkan 软件设备

```bash
vulkaninfo --summary
```

成功输出应包含一个 Vulkan 设备。测试平台上的关键字段类似：

```text
deviceType = PHYSICAL_DEVICE_TYPE_CPU
deviceName = llvmpipe (LLVM 19.1.7, 256 bits)
driverName = llvmpipe
```

`llvmpipe` 和 `PHYSICAL_DEVICE_TYPE_CPU` 表示 shader 正在由 Mesa 软件驱动在
CPU 上执行，正是无 GPU 测试环境需要的结果。如果输出包含
`DISPLAY environment variable not set ... skipping surface info`，只表示跳过窗口
surface 信息；对于本项目的离屏 compute 测试没有影响。

还可以确认 Mesa 的 Vulkan ICD 文件已经安装：

```bash
find /usr/share/vulkan/icd.d -maxdepth 1 -type f -print
```

通常会看到名称包含 `lvp` 的 JSON 文件。一般不需要手动设置 ICD 环境变量；
只有机器存在多个驱动且自动选择错误时，才需要进一步限制驱动。

## 6. 安装并验证 wgpu-py

在平台已有的 Python 环境中安装本测试依赖：

```bash
python -m pip install -r /workspace/tests/requirements.txt
```

如果评测系统把测试文件单独挂载在 `/test_files`，则使用：

```bash
python -m pip install -r /test_files/requirements.txt
```

先单独验证 `wgpu-py` 能枚举 Vulkan adapter：

```bash
WGPU_BACKEND_TYPE=Vulkan python - <<'PY'
import pprint
import wgpu

adapters = wgpu.gpu.enumerate_adapters_sync()
print("adapter count:", len(adapters))
assert adapters, "No WebGPU adapter found"

for adapter in adapters:
    pprint.pprint(adapter.info)

device = adapters[0].request_device_sync()
print("device created:", device is not None)
PY
```

测试平台上的成功结果类似：

```text
adapter count: 1
vendor='llvmpipe'
device='llvmpipe (LLVM 19.1.7, 256 bits)'
adapter_type='CPU'
backend_type='Vulkan'
device created: True
```

这里 `CPU` 只描述 adapter 类型；只要 `backend_type='Vulkan'` 且 device 创建
成功，WGSL compute pipeline 就可以执行。

## 7. 运行本项目测试

测试代码和项目位于同一个 `/workspace` 时：

```bash
cd /workspace
WGPU_BACKEND_TYPE=Vulkan python -m pytest tests -q -s
```

测试代码由平台单独挂载到 `/test_files` 时：

```bash
cd /workspace
WGPU_BACKEND_TYPE=Vulkan python -m pytest /test_files -q -s
```

当前测试会读取真实项目文件：

```text
/workspace/src/renderer/filteredParticleFluid/shader/computeNormal.wgsl
```

测试 fixture 会优先查找 `/workspace`。如果平台使用了其他项目目录，通过
`PROJECT_ROOT` 显式指定，例如：

```bash
PROJECT_ROOT=/another/workspace \
WGPU_BACKEND_TYPE=Vulkan \
python -m pytest /test_files -q -s
```

然后创建 `r32float` 深度纹理、执行 compute pass、读回 `rgba16float` 法向
纹理并与 CPU 参考结果比较。整个过程不创建窗口，也不依赖浏览器。

## 常见问题

### `sudo: command not found`

容器已经是 root，直接执行 `apt-get ...`。

### `Unable to locate package`

先执行 `apt-get update`。如果更新输出中没有 Debian 仓库，再检查或恢复 APT
source；不要通过安装 X Server 解决这个问题。

### `XDG_RUNTIME_DIR is invalid or not set`

按第 4 节创建权限为 `700` 的临时目录并导出 `XDG_RUNTIME_DIR`。即使
`vulkaninfo` 已经列出了设备，这条警告也与 compute shader 的正确性无关。

### `vulkaninfo` 能看到 llvmpipe，但 wgpu-py 没有 adapter

确认以下三点：

```bash
python -c 'import wgpu; print(wgpu.__version__)'
WGPU_BACKEND_TYPE=Vulkan python -c 'import wgpu; print(wgpu.gpu.enumerate_adapters_sync())'
find /usr/share/vulkan/icd.d -maxdepth 1 -type f -print
```

本项目验证过的版本是 `wgpu 0.32.0`。还要确保设置 backend 的环境变量是在
Python 进程启动前传入，而不是在 `import wgpu` 之后才设置。
