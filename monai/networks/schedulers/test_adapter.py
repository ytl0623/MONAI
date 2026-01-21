import torch
import unittest
import diffusers

# 引入你剛剛修改好的 MONAI 組件
# 如果你的修改正確，這裡應該能順利 import
try:
    from monai.networks.schedulers import DiffusersSchedulerAdapter
    from monai.inferers import LatentDiffusionInferer
    print("✅ 成功 import DiffusersSchedulerAdapter")
except ImportError as e:
    print(f"❌ Import 失敗，請檢查檔案位置與 __init__.py: {e}")
    exit(1)

class TestDiffusersAdapter(unittest.TestCase):
    
    def setUp(self):
        # 1. 準備一個 Hugging Face 的 Scheduler
        self.hf_scheduler = diffusers.DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.0015,
            beta_end=0.0205,
            beta_schedule="scaled_linear",
            prediction_type="epsilon",
            clip_sample=False
        )
        
        # 2. 使用你的 Adapter 包裝它
        self.adapter = DiffusersSchedulerAdapter(self.hf_scheduler)
        
        # 3. 準備測試資料 (Batch=1, Channel=4, H=16, W=16)
        self.sample = torch.randn(1, 4, 16, 16)
        self.model_output = torch.randn(1, 4, 16, 16)
        self.timestep = 999

    def test_step_return_type(self):
        """測試 step 是否回傳 tuple (這是 MONAI 最需要的修正)"""
        print("\n[測試 1] 檢查 step() 回傳類型...")
        
        # 呼叫 step
        output = self.adapter.step(self.model_output, self.timestep, self.sample)
        
        # 驗證回傳類型是否為 tuple
        self.assertIsInstance(output, tuple, "錯誤：回傳值應該要是 Tuple")
        self.assertEqual(len(output), 2, "錯誤：Tuple 長度應該為 2 (prev_sample, pred_original_sample)")
        
        print("   -> Pass! 回傳的是 Tuple，MONAI 應該看得懂。")

    def test_attribute_forwarding(self):
        """測試是否能讀取原始 diffusers scheduler 的屬性"""
        print("\n[測試 2] 檢查屬性轉發 (Attribute Forwarding)...")
        
        # 測試讀取 num_train_timesteps
        try:
            timesteps = self.adapter.num_train_timesteps
            self.assertEqual(timesteps, 1000)
            print("   -> Pass! 成功讀取 num_train_timesteps")
        except AttributeError:
            self.fail("無法讀取原始屬性，__getattr__ 可能沒寫好")

    def test_integration_with_inferer(self):
        """測試是否能真的放入 LatentDiffusionInferer 跑完流程"""
        print("\n[測試 3] 整合測試：放入 MONAI LatentDiffusionInferer...")

        # 1. 定義假的擴散模型 (負責去噪)
        class DummyDiffusionModel(torch.nn.Module):
            def forward(self, x, timesteps, context=None):
                return torch.randn_like(x) # 回傳雜訊即可

        # 2. 定義假的 Autoencoder (負責解碼) <--- 這是原本漏掉的部分
        class DummyAutoencoder(torch.nn.Module):
            def decode_stage_2_outputs(self, x):
                # 這裡只要模擬「解碼」動作，直接回傳輸入即可
                # 實際上這裡通常會把 (B, 4, 16, 16) 變成 (B, 3, 128, 128) 之類的，但測試不重要
                return x

        model = DummyDiffusionModel()
        autoencoder = DummyAutoencoder()
        
        # 建立 Inferer
        inferer = LatentDiffusionInferer(scheduler=self.adapter, scale_factor=1.0)
        
        # 模擬輸入雜訊
        noise = torch.randn(1, 4, 16, 16)
        
        try:
            # 設定只要跑 2 步 (節省測試時間)
            self.adapter.set_timesteps(num_inference_steps=2)
            
            # 執行取樣
            generated_image = inferer.sample(
                input_noise=noise,
                diffusion_model=model,
                autoencoder_model=autoencoder,  # <--- 這裡補上了！
                scheduler=self.adapter,
                verbose=False
            )
            
            self.assertIsInstance(generated_image, torch.Tensor)
            print("   -> Pass! Inferer 成功跑完取樣流程 (含 Autoencoder 解碼)！")
            
        except Exception as e:
            self.fail(f"整合測試失敗，Inferer 執行過程中崩潰：{e}")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

