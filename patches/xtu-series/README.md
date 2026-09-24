# vLLM-XTU 补丁系列 —— 从唯一树的提交生成,可直接打到 origin/main
# 生成方式:git -C <vllm-repo> format-patch <上游基线>..<我们的分支> -o patches/xtu-series --no-signature
# 重建方式:每次 rebase 后重新生成,避免手工 patch 腐化(本次实测 16/16 干净应用 ✓)
# 应用:scripts/apply_xtu_patches.sh <含 vllm/ 的目录>(默认即用本系列)
