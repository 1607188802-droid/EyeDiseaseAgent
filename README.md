# 眼科AI辅诊Agent - 病理性近视检测系统

基于ResNet18深度学习模型和DeepSeek大语言模型的医疗辅助诊断系统。

## 功能特点

- 🔬 **眼底图像诊断**：自动识别病理性近视，准确率98%
- 💬 **智能问答**：基于DeepSeek大模型，回答患者问题
- 📊 **置信度评估**：给出诊断置信度
- 🏥 **就诊建议**：提供专业的医疗建议

## 技术架构

- **图像识别**：ResNet18（预训练 + 微调）
- **数据集**：iChallenge-PM（400张训练，400张验证）
- **大模型**：DeepSeek API
- **框架**：PyTorch + OpenAI SDK

## 安装使用

### 1. 安装依赖
```bash
pip install torch torchvision pillow openai python-dotenv