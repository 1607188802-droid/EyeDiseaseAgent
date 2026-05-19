import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import os
from openai import OpenAI

# ========== 配置 DeepSeek API ==========
try:
    from config import DEEPSEEK_API_KEY
except ImportError:
    print(" 请创建 config.py 文件，并设置 DEEPSEEK_API_KEY")
    print("   config.py 内容：")
    print('   DEEPSEEK_API_KEY = "sk-你的key"')
    exit(1)

# 初始化客户端
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# ========== 诊断工具 ==========
class EyeDiagnosisTool:
    def __init__(self, weights_path="best_resnet18.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"初始化诊断工具 - 使用设备: {self.device}")

        if not os.path.exists(weights_path):
            print(f"模型文件不存在: {weights_path}")
            print("   请先运行 train.py 训练模型")
            exit(1)

        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.labels = {
            0: "正常/高度近视 (非病理性)",
            1: "病理性近视"
        }

    def diagnose(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            return {
                "success": True,
                "disease": self.labels[predicted.item()],
                "is_pathologic": predicted.item() == 1,
                "confidence": confidence.item(),
                "prob_non_pathologic": probabilities[0][0].item(),
                "prob_pathologic": probabilities[0][1].item()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# ========== DeepSeek Agent ==========
class MedicalAgentDeepSeek:
    def __init__(self):
        self.diagnosis_tool = EyeDiagnosisTool("best_resnet18.pth")
        self.conversation_history = []

        self.system_prompt = """你是一个专业的眼科AI辅诊助手，名为"EyeDoc-Agent"。

## 你的能力
1. 你可以分析眼底照片，判断是否为病理性近视
2. 你会收到模型的诊断结果（疾病类型、置信度）
3. 你需要用通俗易懂的语言向患者解释病情

## 回答规则
- 如果诊断是"病理性近视"，要说明严重性，建议尽快就医
- 如果诊断是"非病理性"，要安抚患者，给出预防建议
- 解释医学术语时要用比喻（如"漆裂纹"可以说成"眼底像裂开的瓷纹"）
- 语气要温和、专业、有同理心

## 重要提醒
你是一个辅助工具，不能替代真实医生的诊断。在回答末尾加上："*本分析仅供参考，请以医生诊断为准。*"
"""

    def _call_deepseek(self, user_question, diagnosis_result):
        if not diagnosis_result["success"]:
            return f"诊断失败: {diagnosis_result['error']}"

        diagnosis_context = f"""
## 模型诊断结果
- 疾病类型: {diagnosis_result['disease']}
- 是否病理性: {'是' if diagnosis_result['is_pathologic'] else '否'}
- 置信度: {diagnosis_result['confidence']:.1%}
- 病理性概率: {diagnosis_result['prob_pathologic']:.1%}
- 非病理性概率: {diagnosis_result['prob_non_pathologic']:.1%}
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""
用户上传了眼底照片，模型分析结果如下：
{diagnosis_context}

用户的问题：{user_question}

请根据诊断结果回答用户的问题，要专业且易懂。
"""}
        ]

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek 调用失败: {e}")
            # 备用回答
            disease = diagnosis_result['disease']
            if diagnosis_result['is_pathologic']:
                return f"根据模型分析，诊断为{disease}（置信度{diagnosis_result['confidence']:.1%}）。建议尽快到眼科就诊。\n\n*本分析仅供参考，请以医生诊断为准。*"
            else:
                return f"根据模型分析，诊断为{disease}（置信度{diagnosis_result['confidence']:.1%}）。目前看来没有病理性改变，建议定期复查。\n\n*本分析仅供参考，请以医生诊断为准。*"

    def process(self, image_path, user_question):
        diagnosis_result = self.diagnosis_tool.diagnose(image_path)
        answer = self._call_deepseek(user_question, diagnosis_result)

        return {
            "diagnosis": diagnosis_result.get("disease"),
            "confidence": diagnosis_result.get("confidence"),
            "answer": answer
        }


# ========== 主程序 ==========
def main():
    print("\n" + "=" * 60)
    print("眼科AI辅诊Agent（DeepSeek版）")
    print("=" * 60)

    # 检查 API Key
    if DEEPSEEK_API_KEY == "sk-your-deepseek-key-here":
        print("请先配置 DeepSeek API Key！")
        print("   1. 访问 https://platform.deepseek.com/")
        print("   2. 获取 API Key")
        print("   3. 修改代码中的 DEEPSEEK_API_KEY 变量")
        return

    agent = MedicalAgentDeepSeek()

    print("\n使用说明：")
    print("  - 输入图片路径（如: D:\\SqueezeNet\\data\\PALM-Validation400\\V0003.jpg）")
    print("  - 输入您的问题（可以问任何相关问题）")
    print("  - 输入 'quit' 退出\n")

    while True:
        image_path = input("图片路径: ").strip().strip('"')
        if image_path.lower() == 'quit':
            break

        if not os.path.exists(image_path):
            print(f"文件不存在\n")
            continue

        question = input("您的问题: ").strip()
        if question.lower() == 'quit':
            break

        print("\n DeepSeek 分析中...\n")
        result = agent.process(image_path, question)

        print("-" * 50)
        print(f"诊断: {result['diagnosis']} (置信度: {result['confidence']:.1%})")
        print("-" * 50)
        print(f"回答:\n{result['answer']}")
        print("-" * 50)
        print()


if __name__ == "__main__":
    main()