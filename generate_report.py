import json
import pickle
from pathlib import Path
from struct import unpack
import gzip

import numpy as np
import mynn as nn
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak


ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figs"
REPORT = ROOT / "project_outputs" / "MNIST_project_report_李星_22300290002.pdf"
MODEL_FILES = [
    ("mlp_sgd", "MLP + SGD", ROOT / "saved_models" / "best_model_mlp_sgd.pickle"),
    ("cnn_sgd", "CNN + SGD", ROOT / "saved_models" / "best_model_cnn_sgd.pickle"),
    ("cnn_momentum_multistep", "CNN + Momentum + MultiStepLR", ROOT / "saved_models" / "best_model_cnn_momentum_multistep.pickle"),
    ("cnn_l2", "CNN + L2", ROOT / "saved_models" / "best_model_cnn_l2.pickle"),
]


def register_font():
    for path in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont("CN", path))
            return "CN"
    return "Helvetica"


def load_model(path):
    with open(path, "rb") as f:
        state = pickle.load(f)
    model_type = state.get("model_type") if isinstance(state, dict) else "MLP"
    model = nn.models.Model_CNN() if model_type == "CNN" else nn.models.Model_MLP()
    model.load_model(path)
    return model


def load_test_data():
    with gzip.open(ROOT / "dataset" / "MNIST" / "t10k-images-idx3-ubyte.gz", "rb") as f:
        _, num, rows, cols = unpack(">4I", f.read(16))
        imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    with gzip.open(ROOT / "dataset" / "MNIST" / "t10k-labels-idx1-ubyte.gz", "rb") as f:
        _, num = unpack(">2I", f.read(8))
        labs = np.frombuffer(f.read(), dtype=np.uint8)
    return imgs.astype(np.float64) / 255.0, labs


def load_valid_results():
    path = ROOT / "logs" / "train_results.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["name"]: row["best_valid_acc"] for row in rows}


def evaluate_models():
    test_imgs, test_labs = load_test_data()
    loss_fn = nn.op.MultiCrossEntropyLoss(model=None, max_classes=10)
    valid = load_valid_results()
    results = []
    for key, label, path in MODEL_FILES:
        model = load_model(path)
        logits = model(test_imgs)
        results.append({
            "key": key,
            "label": label,
            "valid_acc": valid.get(key),
            "test_acc": float(nn.metric.accuracy(logits, test_labs)),
            "test_loss": float(loss_fn(logits, test_labs)),
            "path": str(path),
        })
    return results


def add_image(story, path, width=15.2 * cm, ratio=0.48):
    if Path(path).exists():
        story.append(Image(str(path), width=width, height=width * ratio))
        story.append(Spacer(1, 0.25 * cm))


def fmt(x):
    return "" if x is None else f"{x:.4f}"


def build():
    REPORT.parent.mkdir(exist_ok=True)
    font = register_font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleCN", parent=styles["Title"], fontName=font, fontSize=21, leading=28, alignment=TA_CENTER))
    styles.add(ParagraphStyle("H1CN", parent=styles["Heading1"], fontName=font, fontSize=15, leading=20, textColor=colors.HexColor("#1f4e79"), spaceBefore=10, spaceAfter=7))
    styles.add(ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=16, spaceAfter=5))
    styles.add(ParagraphStyle("CenterCN", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=16, alignment=TA_CENTER))

    results = evaluate_models()
    best = max(results, key=lambda r: r["test_acc"])

    doc = SimpleDocTemplate(str(REPORT), pagesize=A4, rightMargin=1.8 * cm, leftMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    story = []
    story.append(Paragraph("Project 1：MNIST 手写数字分类", styles["TitleCN"]))
    story.append(Paragraph("神经网络与深度学习", styles["CenterCN"]))
    story.append(Paragraph("姓名：李星　　学号：22300290002", styles["CenterCN"]))
    story.append(Spacer(1, 0.55 * cm))

    story.append(Paragraph("摘要", styles["H1CN"]))
    story.append(Paragraph(
        "本项目基于课程提供的 NumPy starter code，实现了线性层前向/反向传播、softmax 交叉熵、二维卷积、Momentum 优化器和 MultiStepLR 学习率调度器。"
        "实验只使用课程提供的 MNIST 数据集：从 60000 张训练图像中划出 10000 张作为验证集，其余 50000 张用于训练，并在完整 10000 张测试图像上评估。"
        "训练入口为 test_train.py，测试入口为 test_model.py；模型统一保存到 saved_models。",
        styles["BodyCN"]))

    story.append(Paragraph("实验设置", styles["H1CN"]))
    story.append(Paragraph(
        "test_train.py 中按同一数据划分顺序训练四组模型：MLP+SGD、CNN+SGD、CNN+Momentum+MultiStepLR、CNN+L2。"
        "每组都调用 runner.train(..., save_dir=r'./saved_models')，训练结束后将该组最佳模型复制为独立 pickle；最后按验证集准确率选出总最佳模型并保存为 saved_models/best_model.pickle。",
        styles["BodyCN"]))

    story.append(Paragraph("主要结果", styles["H1CN"]))
    rows = [["模型", "最佳验证准确率", "完整测试集准确率", "测试损失"]]
    for r in results:
        rows.append([r["label"], fmt(r["valid_acc"]), fmt(r["test_acc"]), fmt(r["test_loss"])])
    table = Table(rows, colWidths=[6.6 * cm, 3.0 * cm, 3.1 * cm, 2.7 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf7")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365d")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b7b7b7")),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"测试集表现最好的模型为 {best['label']}，完整测试集准确率为 {best['test_acc']:.4f}。", styles["BodyCN"]))

    story.append(Paragraph("学习曲线与可视化", styles["H1CN"]))
    story.append(Paragraph("四个模型的训练曲线分别绘制。每张图左侧为 loss，右侧为 accuracy。", styles["BodyCN"]))
    for curve_name in [
        "mlp_sgd_learning_curve.png",
        "cnn_sgd_learning_curve.png",
        "cnn_momentum_multistep_learning_curve.png",
        "cnn_l2_learning_curve.png",
    ]:
        add_image(story, FIGS / curve_name, width=15.2 * cm, ratio=0.45)

    story.append(PageBreak())
    story.append(Paragraph("MLP baseline", styles["H1CN"]))
    story.append(Paragraph(
        "MLP 使用 784-600-10 结构。输入层对应展平后的 28×28 图像，隐藏层有 600 个神经元，输出层对应 10 个类别。"
        "该模型能够作为基础对照，但由于直接处理展平像素，没有显式利用图像的局部空间结构。",
        styles["BodyCN"]))

    story.append(Paragraph("CNN 模型与 MLP 对比", styles["H1CN"]))
    story.append(Paragraph(
        "CNN 使用自实现 conv2D 提取局部笔画模式，并通过共享卷积核减少参数量。完整测试结果显示 CNN 系列整体优于 MLP baseline，说明卷积结构更适合 MNIST 图像分类。",
        styles["BodyCN"]))

    story.append(Paragraph("附加方向一：优化策略", styles["H1CN"]))
    story.append(Paragraph(
        "优化方向实现了 MomentGD 与 MultiStepLR。Momentum 累积历史梯度方向以减小更新震荡，MultiStepLR 在指定 step 后降低学习率。"
        "从结果看，CNN+Momentum+MultiStepLR 是所有实验中表现最好的模型。",
        styles["BodyCN"]))

    story.append(Paragraph("附加方向二：L2 正则化", styles["H1CN"]))
    story.append(Paragraph(
        "正则方向使用 weight decay 实现 L2 正则。CNN+L2 的表现优于 CNN+SGD，但略低于 Momentum+MultiStepLR，说明正则化有助于泛化，但本实验中优化策略带来的提升更明显。",
        styles["BodyCN"]))

    story.append(Paragraph("详细可视化", styles["H1CN"]))
    story.append(Paragraph("混淆矩阵、错分类样例和权重/卷积核可视化如下。", styles["BodyCN"]))
    add_image(story, FIGS / "confusion_matrix.png", width=12.0 * cm, ratio=1.0)
    add_image(story, FIGS / "misclassified_examples.png", width=12.5 * cm, ratio=1.0)
    add_image(story, FIGS / "weight_or_kernel_visualization.png", width=10.0 * cm, ratio=1.08)

    story.append(Paragraph("讨论", styles["H1CN"]))
    story.append(Paragraph(
        "CNN 更适合图像任务，因为局部连接和权重共享能够有效描述笔画、边缘等局部模式。MLP 作为 baseline 已能达到较高准确率，但 CNN 在相近训练设置下进一步提升了性能。"
        "四组实验中，Momentum+MultiStepLR 的收益最明显；L2 正则也有一定提升，但不是本次实验的最优因素。",
        styles["BodyCN"]))
    story.append(Paragraph("提交代码时不应上传 MNIST 数据集和模型权重；模型权重可上传至 ModelScope 或网盘，并在报告中补充链接。", styles["BodyCN"]))

    doc.build(story)
    print(REPORT)


if __name__ == "__main__":
    build()
