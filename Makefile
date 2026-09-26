# DagForge Makefile  | 作者：晨星
PY ?= python
PIP ?= pip

.PHONY: help install lint test demo ci clean

help:
	@echo "DagForge 命令："
	@echo "  make install  安装依赖（requirements.txt）"
	@echo "  make lint     风格/静态检查（ruff）"
	@echo "  make test     运行测试套件（pytest）"
	@echo "  make demo     生成 benchmark.json"
	@echo "  make ci       本地等价 CI（lint + test + demo 冒烟）"
	@echo "  make clean    清理缓存"

install:
	$(PIP) install -r requirements.txt

lint:
	$(PY) -m py_compile dagforge/*.py dagforge/*/*.py  # 语法/导入门禁
	-$(PY) -m ruff check dagforge tests || true        # 若装了 ruff 则检查

test:
	$(PY) -m pytest -q

demo:
	$(PY) -m dagforge.examples.run_demo --output benchmark.json --seed 42

ci: lint test demo
	@echo "[ok] 本地 CI 等价检查通过"

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache
