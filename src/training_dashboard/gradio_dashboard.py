"""Standalone Gradio UI for automated RL training."""

from __future__ import annotations

import atexit
import json
from typing import Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd

from agent.logger import get_logger

from .agent_loader import AgentLoader
from .config import TrainingDashboardConfig, load_config
from .corpus_manager import CorpusManager, CorpusSummary
from .model_registry import ModelRegistry
from .training_runner import TrainingRequest, TrainingRunner

LOGGER = get_logger(__name__)


def _format_status(status_txt: str, metrics: Dict[str, float]) -> str:
    lines = [status_txt]
    if metrics:
        lines.append("\n**训练指标**")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _summaries_to_table(items: List[CorpusSummary]) -> List[List[str]]:
    table: List[List[str]] = []
    for summary in items:
        table.append([
            summary.path.name,
            str(summary.total),
            summary.description,
            str(summary.path),
        ])
    return table


def build_dashboard(config: Optional[TrainingDashboardConfig] = None) -> gr.Blocks:
    cfg = config or load_config()
    corpus_manager = CorpusManager(cfg)
    corpus_manager.start_scheduler()
    atexit.register(corpus_manager.stop_scheduler)
    registry = ModelRegistry(cfg)
    loader = AgentLoader(cfg)
    runner = TrainingRunner(cfg)

    default_params = cfg.training

    def refresh_status() -> Tuple[str, Dict, str, Optional[pd.DataFrame], List[List[float]]]:
        status = runner.get_status()
        status_txt = (
            f"**状态**: {status.state}\n\n"
            f"{status.message}\n\n"
            f"开始时间: {status.started_at}\n"
            f"结束时间: {status.finished_at}"
        )
        metrics = runner.read_metrics()
        logs = runner.read_log_history()
        series = runner.read_metric_series()
        plot_data: Optional[pd.DataFrame] = None
        table_rows: List[List[float]] = []
        if series:
            plot_data = pd.DataFrame(series)
            table_rows = [[row["step"], row["mean_reward"], row["mean_length"]] for row in series]
        return _format_status(status_txt, metrics), metrics, logs, plot_data, table_rows

    def refresh_logs_live() -> str:
        return runner.read_log_history()

    def refresh_corpus_tables():
        static_table = _summaries_to_table(corpus_manager.list_static_corpus())
        log_table = _summaries_to_table(corpus_manager.list_log_corpus())
        return static_table, log_table

    def ingest_logs_manual() -> str:
        path = corpus_manager.ingest_logs_once()
        if not path:
            return "暂无新增日志"
        return f"已生成: {path}"

    def start_training(
        timesteps: int,
        eval_freq: int,
        checkpoint_freq: int,
        max_steps: int,
        use_embedding: bool,
        use_static: bool,
        use_logs: bool,
    ) -> str:
        if not (use_static or use_logs):
            raise gr.Error("至少需要选择一种语料来源")
        corpus_path, total = corpus_manager.build_training_corpus(use_static, use_logs)
        if not corpus_path or total == 0:
            raise gr.Error("未找到可用语料，请检查语料配置")
        request = TrainingRequest(
            timesteps=int(timesteps),
            eval_freq=int(eval_freq),
            checkpoint_freq=int(checkpoint_freq),
            max_steps_per_episode=int(max_steps),
            use_text_embedding=use_embedding,
            output_dir=default_params.output_dir,
            scenario_file=str(corpus_path),
        )
        runner.start(request)
        return f"训练已启动，语料 {total} 条，输出目录 {request.output_dir}"

    def stop_training() -> str:
        runner.stop()
        return "已请求停止训练"

    def list_models() -> List[List[str]]:
        rows: List[List[str]] = []
        for entry in registry.list_models():
            metadata = ""
            if entry.metadata_path.exists():
                metadata = entry.metadata_path.read_text(encoding="utf-8")
            rows.append([
                entry.model_id,
                "✅" if entry.best_model and entry.best_model.exists() else "",
                "✅" if entry.final_model and entry.final_model.exists() else "",
                metadata,
            ])
        return rows

    def apply_model(model_id: str, variant: str) -> str:
        entry = registry.get_entry(model_id)
        if not entry:
            raise gr.Error("未找到模型记录")
        target = loader.apply_model(entry, variant)
        return f"模型 {model_id} 已同步到 {target}"

    with gr.Blocks(css="""#logs {min-height:200px;}""") as demo:
        gr.Markdown("# 🧠 RL 训练控制台")
        with gr.Tab("概览"):
            status_md = gr.Markdown("尚未启动", elem_id="status_panel")
            with gr.Row():
                metrics_json = gr.JSON(label="最新训练指标")
                metrics_plot = gr.LinePlot(
                    label="奖励/长度走势",
                    x="step",
                    y=["mean_reward", "mean_length"],
                    title="训练过程指标",
                    tooltip=["mean_reward", "mean_length"],
                    height=320,
                )
            metrics_table = gr.Dataframe(
                headers=["Step", "Mean Reward", "Mean Length"],
                interactive=False,
                wrap=True,
                label="详细指标",
            )
            logs_box = gr.Textbox(label="训练日志", lines=10, elem_id="logs")
            log_timer = gr.Timer(value=3.0)
            log_timer.tick(refresh_logs_live, outputs=logs_box)
            refresh_btn = gr.Button("刷新状态")
            refresh_btn.click(
                refresh_status,
                outputs=[status_md, metrics_json, logs_box, metrics_plot, metrics_table],
            )
            stop_btn = gr.Button("停止训练", variant="stop")
            stop_result = gr.Markdown()
            stop_btn.click(stop_training, outputs=stop_result)

        with gr.Tab("语料管理"):
            ingest_info = gr.Markdown("日志提炼任务按配置自动运行，也可手动触发。")
            static_table = gr.Dataframe(headers=["文件", "场景数", "类型", "路径"], interactive=False)
            log_table = gr.Dataframe(headers=["文件", "场景数", "类型", "路径"], interactive=False)
            refresh_corpus_btn = gr.Button("刷新语料列表")
            refresh_corpus_btn.click(refresh_corpus_tables, outputs=[static_table, log_table])
            ingest_btn = gr.Button("手动提炼日志")
            ingest_result = gr.Markdown()
            ingest_btn.click(ingest_logs_manual, outputs=ingest_result)

        with gr.Tab("训练控制"):
            with gr.Row():
                timesteps = gr.Number(label="训练步数", value=default_params.timesteps)
                eval_freq = gr.Number(label="评估频率", value=default_params.eval_freq)
                checkpoint_freq = gr.Number(label="检查点频率", value=default_params.checkpoint_freq)
                max_steps = gr.Number(label="Episode 步长", value=default_params.max_steps_per_episode)
            use_embedding = gr.Checkbox(label="开启文本嵌入", value=default_params.use_text_embedding)
            use_static = gr.Checkbox(label="使用静态语料", value=True)
            use_logs = gr.Checkbox(label="使用日志语料", value=True)
            start_btn = gr.Button("启动训练", variant="primary")
            start_status = gr.Markdown()
            start_btn.click(
                start_training,
                inputs=[
                    timesteps,
                    eval_freq,
                    checkpoint_freq,
                    max_steps,
                    use_embedding,
                    use_static,
                    use_logs,
                ],
                outputs=start_status,
            )

        with gr.Tab("模型管理"):
            model_table = gr.Dataframe(
                headers=["模型ID", "Best", "Final", "元数据"],
                interactive=False,
            )
            refresh_models_btn = gr.Button("刷新模型列表")
            refresh_models_btn.click(list_models, outputs=model_table)
            model_id_box = gr.Textbox(label="模型ID")
            variant_radio = gr.Radio(["best", "final"], label="加载版本", value="best")
            apply_btn = gr.Button("加载到 Agent")
            apply_result = gr.Markdown()
            apply_btn.click(apply_model, inputs=[model_id_box, variant_radio], outputs=apply_result)

        demo.load(refresh_corpus_tables, outputs=[static_table, log_table])
        demo.load(
            refresh_status,
            outputs=[status_md, metrics_json, logs_box, metrics_plot, metrics_table],
        )
        demo.load(list_models, outputs=model_table)
    return demo


def launch() -> gr.Blocks:
    demo = build_dashboard()
    demo.queue()
    demo.launch()
    return demo
