# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (License header) ...

from __future__ import annotations

from typing import Any

# 使用 MONAI 的工具來安全匯入 diffusers，避免使用者沒安裝時報錯
from monai.utils import optional_import

# 嘗試匯入 diffusers，如果沒安裝，has_diffusers 會是 False
diffusers, has_diffusers = optional_import("diffusers")
SchedulerMixin, _ = optional_import("diffusers.schedulers.scheduling_utils", name="SchedulerMixin")

class DiffusersSchedulerAdapter(SchedulerMixin if has_diffusers else object):
    """
    Adapter to make Hugging Face diffusers schedulers compatible with MONAI inferers.
    It wraps a diffusers scheduler and forces `return_dict=False` in the `step` method.
    """

    def __init__(self, scheduler: Any) -> None:
        if not has_diffusers:
            raise ImportError("Please install the 'diffusers' library to use this adapter.")
        
        self.scheduler = scheduler

    def step(self, model_output, timestep, sample, **kwargs):
        # 攔截並強制轉為 Tuple 輸出
        return self.scheduler.step(
            model_output, 
            timestep, 
            sample, 
            return_dict=False, 
            **kwargs
        )

    def __getattr__(self, name: str):
        # 將其他所有屬性呼叫轉發給原始 scheduler
        return getattr(self.scheduler, name)

