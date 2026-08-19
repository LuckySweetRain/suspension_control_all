---
name: verify
description: 运行本仓库可用的验证手段（无 pytest 套件时的替代）。校验训练配置结构、Gymnasium 环境注册，并做关键模块导入冒烟检查。在改动配置、环境或控制器后、标记任务完成前使用。
---

# /verify

本仓库没有 pytest 测试套件，本技能是最接近的验证手段。接受一个可选参数作为配置路径，默认 `configs/default.yaml`。

## 步骤

1. **ruff lint 检查**（快速，捕获语法错误、导入问题、bugbear、pyupgrade 建议）：
   ```
   ruff check
   ```
   从仓库根运行，读取 `pyproject.toml` 配置。若有可自动修复的问题，可提示用户运行 `ruff check --fix`；不要自动执行。

2. **配置结构校验**（校验 YAML 结构、obs/act 维度、配置一致性，写 JSON 报告）：
   ```
   python scripts/validate_training_config.py --config $ARGUMENTS --out results/verify_config.json
   ```
   若 `$ARGUMENTS` 为空，用 `configs/default.yaml`。读 `results/verify_config.json` 确认 `valid: true`。

3. **环境注册 + 冒烟检查**（注册 Gymnasium 环境并尝试创建实例）：
   ```
   python scripts/check_gym_registration.py --config $ARGUMENTS --smoke
   ```
   确认输出含 `gymnasium_available: true` 且冒烟 `smoke` 通过。

4. **关键模块导入冒烟检查**（捕获语法/导入错误，尤其改过控制器或环境后）：
   ```
   python -c "import controllers; from controllers import PIDController, SPDFController, MPCController, RLTransformerController; from models.half_car import HalfCarModel; from envs import register_gymnasium_envs; register_gymnasium_envs(); print('imports ok')"
   ```
   必须从仓库根目录运行（脚本依赖根在 `sys.path`）。

## 判定

任一步失败即报告失败原因并停止，不要标记后续任务完成。全部通过则简述结果。

## 注意

- 若 `$ARGUMENTS` 指向继承链中的子配置，校验会按 `inherits:` 解析完整配置。
- 不要运行 `pytest`——本仓库已移除测试套件，运行只会报 "no tests"。
