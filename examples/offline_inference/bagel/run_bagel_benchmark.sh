#!/bin/bash
# BAGEL Text-to-Image Benchmark Script
# 用于测试 BAGEL 模型文生图的吞吐量和延迟

set -o pipefail  # 管道中任意命令失败则管道返回失败
# 注意：不使用 set -e，避免 benchmark 失败时提前退出并杀掉服务器

# ============== 配置参数 ==============
MODEL_PATH="/mnt/ceph-hz1-csp/yunhaohan/models/BAGEL-7B-MoT"
STAGE_CONFIG="bagel_h20.yaml"
SERVER_PORT=8099
SERVER_HOST="0.0.0.0"
BENCHMARK_URL="http://127.0.0.1:${SERVER_PORT}"  # benchmark 用固定 IPv4 地址

# Benchmark 参数
NUM_PROMPTS=10           # 测试请求数量
MAX_CONCURRENCY=2        # 最大并发数
IMAGE_WIDTH=1024         # 生成图像宽度
IMAGE_HEIGHT=1024        # 生成图像高度
NUM_INFERENCE_STEPS=50   # 推理步数
WARMUP_REQUESTS=1        # 预热请求数

# GPU 配置
# 注意：不要设置 CUDA_VISIBLE_DEVICES，让 stage config 中的 devices 设置生效
# 如果需要指定 GPU，请在 bagel_h20.yaml 中修改 devices 字段
# export CUDA_VISIBLE_DEVICES=0,1
GPU_IDS="0,1"                # 需要监控的 GPU ID（逗号分隔，与 stage config 中 devices 保持一致）
ENABLE_GPU_MONITOR=true      # 是否启用 GPU SM 利用率监控

# 输出目录（使用绝对路径，避免 cd 后路径失效）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/benchmark_results"
mkdir -p ${OUTPUT_DIR}

# ============== 颜色输出 ==============
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============== 清理函数 ==============
cleanup() {
    log_info "Cleaning up..."
    # 停止 GPU 监控
    stop_gpu_monitor 2>/dev/null || true
    if [ ! -z "$SERVER_PID" ]; then
        log_info "Stopping server (PID: $SERVER_PID) and all child processes..."
        # 先用 SIGTERM 优雅关闭，给进程 5 秒清理时间
        kill -TERM -- -$SERVER_PID 2>/dev/null || kill -TERM $SERVER_PID 2>/dev/null || true
        sleep 2
        # 再用 SIGKILL 强制杀死残留进程
        kill -9 -- -$SERVER_PID 2>/dev/null || kill -9 $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    # 兜底：清理所有匹配的残留进程
    pkill -9 -f "vllm serve.*${MODEL_PATH}" 2>/dev/null || true
    # 杀死 vllm 的子进程（VLLM::Worker, VLLM::EngineCor, resource_tracker 等）
    pkill -9 -f "VLLM::" 2>/dev/null || true
    # 等待子进程完全退出
    sleep 2
    # 通过 fuser 清理 GPU 占用的残留进程
    local gpu_id
    IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
    for gpu_id in "${GPU_ARRAY[@]}"; do
        local gpu_pids
        gpu_pids=$(fuser /dev/nvidia${gpu_id} 2>/dev/null || true)
        if [ -n "$gpu_pids" ]; then
            log_warn "GPU ${gpu_id} still has processes: ${gpu_pids}, killing..."
            echo "$gpu_pids" | xargs kill -9 2>/dev/null || true
        fi
    done
}

trap cleanup EXIT

# ============== 检查依赖 ==============
check_dependencies() {
    log_info "Checking dependencies..."
    
    # 检查 vllm 命令
    if ! command -v vllm &> /dev/null; then
        log_error "vllm command not found. Please install vllm-omni first."
        exit 1
    fi
    
    # 检查模型路径
    if [ ! -d "$MODEL_PATH" ]; then
        log_error "Model path not found: $MODEL_PATH"
        exit 1
    fi
    
    # 检查配置文件
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [ ! -f "${SCRIPT_DIR}/${STAGE_CONFIG}" ]; then
        log_error "Stage config not found: ${SCRIPT_DIR}/${STAGE_CONFIG}"
        exit 1
    fi
    
    log_info "All dependencies OK"
}

# ============== 等待服务器启动 ==============
wait_for_server() {
    local max_wait=600  # 最大等待 10 分钟（模型加载需要时间）
    local wait_time=0
    
    log_info "Waiting for server to start (max ${max_wait}s)..."
    
    while [ $wait_time -lt $max_wait ]; do
        # 先检查服务器进程是否还活着
        if [ -n "$SERVER_PID" ] && ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ""
            log_error "Server process (PID: $SERVER_PID) died during startup"
            log_info "Last 50 lines of server log:"
            tail -50 "${OUTPUT_DIR}/server.log" 2>/dev/null || echo "No log available"
            return 1
        fi
        
        # 检查 /health 端点
        if curl -s --connect-timeout 5 "http://127.0.0.1:${SERVER_PORT}/health" > /dev/null 2>&1; then
            echo ""
            log_info "Server health check passed, waiting 15s for engine stabilization..."
            # 等待 15 秒（vllm watchdog 每 5 秒检查一次引擎状态，
            # 如果 engine errored，watchdog 会在 5-10 秒内触发 shutdown）
            sleep 15
            
            # 再次验证服务器仍然存活
            if [ -n "$SERVER_PID" ] && ! kill -0 $SERVER_PID 2>/dev/null; then
                echo ""
                log_error "Server process died after startup (engine may have errored)"
                log_info "Last 50 lines of server log:"
                tail -50 "${OUTPUT_DIR}/server.log" 2>/dev/null || echo "No log available"
                return 1
            fi
            
            # 最终确认 health check 仍然能通过
            if curl -s --connect-timeout 5 "http://127.0.0.1:${SERVER_PORT}/health" > /dev/null 2>&1; then
                log_info "Server is ready and stable!"
                return 0
            else
                log_error "Server health check failed after stabilization wait"
                log_info "Last 50 lines of server log:"
                tail -50 "${OUTPUT_DIR}/server.log" 2>/dev/null || echo "No log available"
                return 1
            fi
        fi
        
        sleep 5
        wait_time=$((wait_time + 5))
        echo -n "."
    done
    
    echo ""
    log_error "Server failed to start within ${max_wait} seconds"
    log_info "Last 50 lines of server log:"
    tail -50 "${OUTPUT_DIR}/server.log" 2>/dev/null || echo "No log available"
    return 1
}

# ============== 启动服务器 ==============
start_server() {
    log_info "Starting vLLM-Omni server..."
    log_info "  Model: ${MODEL_PATH}"
    log_info "  Port: ${SERVER_PORT}"
    log_info "  GPUs: ${CUDA_VISIBLE_DEVICES}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # 启动服务器（后台运行）
    vllm serve ${MODEL_PATH} \
        --omni \
        --stage-configs-path "${SCRIPT_DIR}/${STAGE_CONFIG}" \
        --port ${SERVER_PORT} \
        --host ${SERVER_HOST} \
        > "${OUTPUT_DIR}/server.log" 2>&1 &
    
    SERVER_PID=$!
    log_info "Server started with PID: ${SERVER_PID}"
    
    # 等待服务器就绪
    if ! wait_for_server; then
        log_error "Server startup failed. Check ${OUTPUT_DIR}/server.log for details."
        exit 1
    fi
}

# ============== 运行 Benchmark ==============
GPU_MONITOR_PID=""

# 启动 GPU SM 监控
start_gpu_monitor() {
    local log_file=$1
    if [ "${ENABLE_GPU_MONITOR}" != "true" ]; then
        return
    fi
    log_info "Starting GPU SM monitor (GPUs: ${GPU_IDS}) -> ${log_file}"
    nvidia-smi dmon -i ${GPU_IDS} -s u -d 1 > "${log_file}" 2>/dev/null &
    GPU_MONITOR_PID=$!
}

# 停止 GPU SM 监控
stop_gpu_monitor() {
    if [ -n "${GPU_MONITOR_PID}" ] && kill -0 ${GPU_MONITOR_PID} 2>/dev/null; then
        kill ${GPU_MONITOR_PID} 2>/dev/null || true
        wait ${GPU_MONITOR_PID} 2>/dev/null || true
        GPU_MONITOR_PID=""
        log_info "GPU SM monitor stopped"
    fi
}

# 分析 GPU SM 利用率
analyze_gpu_sm() {
    local log_file=$1
    local report_file=$2
    if [ "${ENABLE_GPU_MONITOR}" != "true" ]; then
        return
    fi
    if [ ! -s "${log_file}" ]; then
        log_warn "GPU SM log is empty, skipping analysis"
        return
    fi
    log_info "Analyzing GPU SM utilization..."
    python3 "${SCRIPT_DIR}/analyze_gpu_sm.py" "${log_file}" \
        --gpus "${GPU_IDS}" \
        --output "${report_file}" \
        2>&1 || log_warn "GPU SM analysis failed"
}

run_benchmark() {
    local test_name=$1
    local num_prompts=$2
    local max_concurrency=$3
    local request_rate=$4
    
    # 确认服务器仍然存活
    if [ -n "$SERVER_PID" ] && ! kill -0 $SERVER_PID 2>/dev/null; then
        log_error "Server process (PID: $SERVER_PID) is not running! Skipping benchmark."
        log_info "Last 30 lines of server log:"
        tail -30 "${OUTPUT_DIR}/server.log" 2>/dev/null || true
        return 1
    fi
    if ! curl -s --connect-timeout 5 "http://127.0.0.1:${SERVER_PORT}/health" > /dev/null 2>&1; then
        log_error "Server health check failed! Server may have crashed."
        log_info "Last 30 lines of server log:"
        tail -30 "${OUTPUT_DIR}/server.log" 2>/dev/null || true
        return 1
    fi
    
    log_info "Running benchmark: ${test_name}"
    log_info "  Prompts: ${num_prompts}, Concurrency: ${max_concurrency}, Rate: ${request_rate}"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="${OUTPUT_DIR}/benchmark_${test_name}_${TIMESTAMP}.json"
    GPU_SM_LOG="${OUTPUT_DIR}/gpu_sm_${test_name}_${TIMESTAMP}.csv"
    GPU_SM_REPORT="${OUTPUT_DIR}/gpu_sm_${test_name}_${TIMESTAMP}_report.txt"
    
    # 启动 GPU SM 监控
    start_gpu_monitor "${GPU_SM_LOG}"
    
    # 获取 vllm-omni 项目根目录（相对于脚本位置向上 3 级）
    local VLLM_OMNI_ROOT
    VLLM_OMNI_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
    cd "${VLLM_OMNI_ROOT}" || { log_error "Cannot cd to vllm-omni root: ${VLLM_OMNI_ROOT}"; return 1; }
    log_info "Working directory: $(pwd)"
    
    python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
        --base-url "${BENCHMARK_URL}" \
        --model "${MODEL_PATH}" \
        --backend vllm-omni \
        --task t2i \
        --dataset vbench \
        --num-prompts ${num_prompts} \
        --max-concurrency ${max_concurrency} \
        --request-rate ${request_rate} \
        --width ${IMAGE_WIDTH} \
        --height ${IMAGE_HEIGHT} \
        --num-inference-steps ${NUM_INFERENCE_STEPS} \
        --warmup-requests ${WARMUP_REQUESTS} \
        --output-file "${OUTPUT_FILE}" \
        2>&1 | tee "${OUTPUT_DIR}/benchmark_${test_name}_${TIMESTAMP}.log" || true
    
    # 停止 GPU SM 监控并生成分析报告
    stop_gpu_monitor
    analyze_gpu_sm "${GPU_SM_LOG}" "${GPU_SM_REPORT}"
    
    log_info "Results saved to: ${OUTPUT_FILE}"
    if [ -f "${GPU_SM_REPORT}" ]; then
        log_info "GPU SM report: ${GPU_SM_REPORT}"
    fi
}

# ============== 主流程 ==============
main() {
    echo ""
    echo "=========================================="
    echo "  BAGEL Text-to-Image Benchmark"
    echo "=========================================="
    echo ""
    
    # 检查依赖
    check_dependencies
    
    # 启动服务器
    start_server
    
    echo ""
    log_info "Starting benchmark tests..."
    echo ""
    
    # ====== 测试 1: 单并发基准测试 ======
    run_benchmark "single_concurrency" ${NUM_PROMPTS} 1 "inf"
    
    # ====== 测试 2: 多并发测试 ======
    # run_benchmark "multi_concurrency_2" ${NUM_PROMPTS} 2 "inf"
    # run_benchmark "multi_concurrency_4" ${NUM_PROMPTS} 4 "inf"
    
    # ====== 测试 3: 固定 QPS 测试 ======
    # run_benchmark "qps_0.5" ${NUM_PROMPTS} 4 0.5
    # run_benchmark "qps_1.0" ${NUM_PROMPTS} 4 1.0
    
    echo ""
    log_info "=========================================="
    log_info "  Benchmark completed!"
    log_info "  Results directory: ${OUTPUT_DIR}"
    log_info "=========================================="
    echo ""
    
    # 显示结果摘要
    log_info "Result files:"
    ls -la ${OUTPUT_DIR}/*.json 2>/dev/null || echo "No JSON results found"
    
    # 显示 GPU SM 分析报告
    if [ "${ENABLE_GPU_MONITOR}" = "true" ]; then
        echo ""
        for report in ${OUTPUT_DIR}/gpu_sm_*_report.txt; do
            if [ -f "${report}" ]; then
                log_info "GPU SM Analysis Report: $(basename ${report})"
                cat "${report}"
                echo ""
            fi
        done
    fi
}

# ============== 仅运行 Benchmark（服务器已启动时使用） ==============
benchmark_only() {
    echo ""
    log_info "Running benchmark only (assuming server is already running)..."
    echo ""
    
    # 检查服务器是否已启动
    if ! curl -s "${BENCHMARK_URL}/health" > /dev/null 2>&1; then
        log_error "Server is not running at ${BENCHMARK_URL}"
        log_info "Please start the server first or use './run_bagel_benchmark.sh' without arguments"
        exit 1
    fi
    
    run_benchmark "single_concurrency" ${NUM_PROMPTS} 1 "inf"
    
    log_info "Benchmark completed!"
}

# ============== 入口 ==============
case "${1:-}" in
    --benchmark-only|-b)
        benchmark_only
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  (no args)             Start server and run benchmark"
        echo "  -b, --benchmark-only  Run benchmark only (server must be running)"
        echo "  -h, --help            Show this help message"
        echo ""
        echo "Configuration (edit script to change):"
        echo "  NUM_PROMPTS           Number of test requests (default: 10)"
        echo "  MAX_CONCURRENCY       Max concurrency (default: 1)"
        echo "  GPU_IDS               GPUs to monitor (default: 0,1)"
        echo "  ENABLE_GPU_MONITOR    Enable GPU SM monitoring (default: true)"
        echo ""
        echo "Output files (in benchmark_results/):"
        echo "  benchmark_*.json      Benchmark metrics"
        echo "  benchmark_*.log       Benchmark console output"
        echo "  gpu_sm_*.csv          Raw GPU SM utilization data"
        echo "  gpu_sm_*_report.txt   GPU SM analysis report"
        ;;
    *)
        main
        ;;
esac
