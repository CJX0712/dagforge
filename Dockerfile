# DagForge 运行镜像（确定性因果发现）
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

WORKDIR /app

# 锁定依赖优先，回退到运行依赖
COPY requirements.lock.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt || pip install --no-cache-dir -r requirements.txt

COPY . .

# 默认跑基准并落盘（可在容器外挂载卷取走 benchmark.json）
ENTRYPOINT ["python", "-m", "dagforge.examples.run_demo", "--output", "benchmark.json", "--seed", "42"]
