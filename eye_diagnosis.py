import to  rch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import os


class EyeDiagnosisTool:
    def __init__(self, weights_path="best_resnet18.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # 加载训练好的 ResNet18 模型
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 图像预处理（必须和训练时一致）
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
        """诊断单张眼底图像"""
        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # 推理
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            disease = self.labels[predicted.item()]
            confidence_score = confidence.item()

            # 获取各类概率
            probs = probabilities.cpu().numpy().tolist()[0]

            return {
                "success": True,
                "disease": disease,
                "confidence": confidence_score,
                "prob_non_pathologic": probs[0],
                "prob_pathologic": probs[1]
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# 测试代码
if __name__ == "__main__":
    tool = EyeDiagnosisTool("best_resnet18.pth")

    # 测试你的三张图片
    test_images = [
        ("test_images/N0002.jpg", "正常"),
        ("test_images/H0002.jpg", "高度近视"),
        ("test_images/P0002.jpg", "病理性近视")
    ]

    print("\n" + "=" * 50)
    for img_path, true_label in test_images:
        if os.path.exists(img_path):
            result = tool.diagnose(img_path)
            if result["success"]:
                print(f"\n📷 图片: {true_label}")
                print(f"   诊断结果: {result['disease']}")
                print(f"   置信度: {result['confidence']:.2%}")
                print(f"   病理性概率: {result['prob_pathologic']:.2%}")
                print(f"   非病理性概率: {result['prob_non_pathologic']:.2%}")
            else:
                print(f"错误: {result['error']}")
        else:
            print(f"文件不存在: {img_path}")