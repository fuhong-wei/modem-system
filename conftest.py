"""pytest 全局配置。"""

import os
import sys

import matplotlib

# 无界面环境下使用 Agg 后端，避免测试时依赖显示器
matplotlib.use("Agg")

# 确保项目根目录在 sys.path 中，便于 ``import src``
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
