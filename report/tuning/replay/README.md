# `report/tuning/replay/` —— 预填充基准用的 prompt 集

## 1. `sharegpt16*.jsonl`(公开数据,可入库)
由 `ShareGPT_V3_unfiltered_cleaned_split.json` 复刻 **`vllm bench serve --dataset-name sharegpt`
实际会取到的那 16 条**(复现其 shuffle(seed=0) + `is_valid_sequence` 过滤);校验点:
`total_input_tokens == 3638`,与官方 loader 口径逐位相同。
`sharegpt16_sorted.jsonl` 是按 token 长度降序(配合 `--disable-shuffle --num-prompts 8` 取"最长的 8 条")。

**为什么需要复刻**:官方 ShareGPT loader 的 `is_valid_sequence` 默认
`max_prompt_len=1024, max_total_len=2048`(`datasets.py:341-347`,ShareGPTDataset 不覆盖它)
⇒ **ShareGPT 口径做不了长 prompt,也做不了 out>2048**。要变长只能走
`--dataset-name custom`(它完全不做该校验)。

## 2. `dsh_*.json` / `dsh_*.jsonl`(**不入库**,见 `.gitignore`)
**不是 ShareGPT** —— 是本机 DSH 会话 transcript(`~/.dsh/sessions/*/session.v3.jsonl.zstd`)
末尾若干轮的**原文回放**,按 token 截成 2K/4K/8K/16K/32K。

用途:ShareGPT 平均 prompt 只有 ~227 token,**永远触发不到 GPU 预填充**
(`VLLM_XIAOTU_GPU_PREFILL_MIN_TOKENS=4096`);而 agentic coding 的真实轮次是几千 token,
才是 GPU 预填充的目标区间。

**现场重建**(内容含用户对话原文,所以只在本机生成):
```bash
python scripts/make_dsh_replay.py \
  --tokenizer /home/user/.cache/modelscope/models/deepseek-ai--DeepSeek-V4.1-Flash/snapshots/master
```

**引用时务必标注**:`DSH 开发会话回放(非 ShareGPT)`。
