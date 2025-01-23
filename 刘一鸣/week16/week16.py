'''
1.实现yolov5
2.实现unet
3.实现deepsort（简单or手写）
4.实现openpose（简单or手写）
'''

#1、实现yolov5
from ultralytics import YOLO
import cv2

# 加载 YOLOv5 模型（预训练权重）
model = YOLO('yolov5s.pt')  # 使用 "s" 模型，适合快速检测

# 读取图片
image_path = 'street.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式

# 进行目标检测
results = model(img)

# 显示检测结果
results[0].plot()  # 绘制检测结果（默认弹窗显示）

# 使用 Matplotlib 显示结果
plt.imshow(results[0].plot())
plt.axis('off')
plt.show()

#2、实现unet
import torch.nn as nn
import torch.nn.functional as F

# 定义 U-Net 模型
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # 下采样部分（编码器）
        self.enc1 = self.conv_block(1, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # 中间部分
        self.bottleneck = self.conv_block(512, 1024)

        # 上采样部分（解码器）
        self.upconv4 = self.upconv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        # 最终输出层
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # 中间
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # 解码器
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)  # 跳跃连接
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # 最终输出
        return torch.sigmoid(self.final(d1))

# 初始化模型
model = UNet()

#3、实现deepsort（简单or手写）
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# 初始化 YOLOv5 模型（用于检测）
model = YOLO('yolov5s.pt')  # 使用 YOLOv5 小型模型

# 初始化 DeepSORT 跟踪器
tracker = DeepSort(max_age=30,  # 最大失去帧数
                   n_init=3,  # 初始确认的帧数
                   nn_budget=100,  # 特征存储上限
                   max_iou_distance=0.7)  # IOU 阈值

# 视频输入
video_path = 'video.mp4'
cap = cv2.VideoCapture(video_path)

# 视频输出
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将帧传入 YOLOv5 检测
    results = model(frame)

    # 获取检测框信息（左上角坐标、宽高、置信度、类别）
    detections = []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = map(int, box)  # 边界框
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), int(cls.item())))

    # 将检测结果传入 DeepSORT 跟踪器
    tracks = tracker.update_tracks(detections, frame=frame)

    # 绘制跟踪结果
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id  # 跟踪 ID
        l, t, w, h = track.to_ltwh().astype(int)  # 边界框
        cv2.rectangle(frame, (l, t), (l + w, t + h), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 写入视频文件
    out.write(frame)

    # 显示结果（可选）
    cv2.imshow('DeepSORT Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

#4、实现openpose（简单or手写）
import openpifpaf
import matplotlib.pyplot as plt

# 加载预训练模型
predictor = openpifpaf.Predictor()

# 读取测试图片
image_path = 'street.jpg'  # 替换为你的图片路径
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 进行姿态估计
predictions, scores, _ = predictor.numpy_image(img_rgb)

# 使用 KeypointPainter 绘制检测结果
from openpifpaf.show import KeypointPainter
painter = KeypointPainter()

# 绘制检测结果
fig, ax = plt.subplots(figsize=(10, 10))
for prediction in predictions:
    painter.annotation(ax, prediction)  # 替换为 'annotation' 方法
ax.imshow(img_rgb)
plt.axis('off')
plt.show()



