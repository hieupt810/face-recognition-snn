import os
import shutil
import time
from uuid import uuid4

import cv2
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(torch.transpose(output, 0, 1))
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


def verify(input_image, user_folder, verify_threshold):
    # Load model
    transformation = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((100, 100)), torchvision.transforms.ToTensor()]
    )
    device = torch.device("cpu")
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load("SNN50.pt", map_location=device))

    # Process
    x1 = transformation(Image.open(input_image).convert("L"))
    threshold = []
    start_time = time.time()

    for anchor in os.listdir(user_folder):
        img = os.path.join(user_folder, anchor)
        x2 = transformation(Image.open(img).convert("L"))

        output1, output2 = model.forward(x1, x2)
        euclidean_distance = F.pairwise_distance(output1, output2)
        threshold.append(euclidean_distance.item())

    return numpy.average(threshold) < verify_threshold, round(
        time.time() - start_time, 2
    )


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)

    anchor_path = os.path.join(os.getcwd(), "user")
    if os.path.isdir(anchor_path):
        shutil.rmtree(anchor_path)

    os.mkdir(anchor_path)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[300:700, 550:850, :]
        cv2.imshow("Real Time Verification", frame)

        if cv2.waitKey(10) & 0xFF == ord("a"):
            img = os.path.join(anchor_path, "{}.jpg".format(uuid4()))
            cv2.imwrite(img, frame)

        if cv2.waitKey(10) & 0xFF == ord("r"):
            if os.path.isdir(anchor_path):
                shutil.rmtree(anchor_path)

            os.mkdir(anchor_path)

        if cv2.waitKey(10) & 0xFF == ord("v"):
            os.system("clear")

            input = os.path.join(os.getcwd(), "input.jpg")
            cv2.imwrite(input, frame)

            value, execution_time = verify(input, anchor_path, 0.6)
            print("Verified!" if value else "Unverified!")
            print("Execution time:", execution_time, "second")

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
