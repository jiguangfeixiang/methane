import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from conda3.Lib.http.client import responses
from hparam_test import hparams as hp
from metric import metrics_01, metrics_cont, mape_test
from data_function import XCh4Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
devicess = [0]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#文件路径
source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir
bin_label_test_dir = hp.bin_label_test_dir
emission_label_test_dir = hp.emission_label_test_dir

#输出路径
output_dir_test = hp.output_dir_test


def plot_emission_rate(predictions, labels):
    # 将列表转换为numpy数组
    predictions = np.array(predictions)
    labels = np.array(labels)

    # 绘制散点图
    plt.scatter(labels, predictions, color='b', alpha=0.1)

    # 在图中添加 45 度虚线
    max_value = max(labels.max(), predictions.max())
    plt.plot([0, max_value], [0, max_value], 'k--', linewidth=1)

    # 计算和绘制最佳拟合线
    slope, intercept = np.polyfit(labels, predictions, 1)
    plt.plot(labels, slope * labels + intercept, color='blue', linewidth=2)

    # 添加斜率值的文本
    plt.text(0.05 * max_value, 0.9 * max_value, f'Slope={slope:.2f}', fontsize=10)

    # 设置轴标签和标题
    plt.xlabel('True emission rate (kg/hr)')
    plt.ylabel('Predicted emission rate (kg/hr)')
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)

    plt.show()


def save_colored_heatmap(image_data, filepath):  # 保存彩色热力图
    """保存彩色热力图，黑色部分变成透明"""

    # 将图像缩小为 75x75
    image_data_resized = zoom(image_data, (75 / 80, 75 / 80), order=1)  # 线性插值

    # 创建一个与 image_data_resized 相同形状的 alpha 通道
    alpha_layer = np.ones(image_data_resized.shape)  # 默认全部不透明

    # 将纯黑部分设置为透明
    alpha_layer[image_data_resized == 0] = 0  # 假设黑色数据为0

    # 绘制热力图并设置 alpha
    plt.imshow(image_data_resized, cmap='hot', alpha=alpha_layer)
    plt.axis('off')  # 不显示坐标轴
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()


def test():
    torch.backends.cudnn.deterministic = True  # 设置 PyTorch 的 cuDNN 后端为确定性模式。在确定性模式下，每次运行相同的输入和相同的操作时，都会得到相同的结果。这对于调试和复现结果非常有用。
    torch.backends.cudnn.enabled = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = torch.cuda.is_available() and False  #没有GPU

    os.makedirs(output_dir_test, exist_ok=True)

    if hp.mode == "2d":
        from models.resnetme import ResNet
        from models.unetme import UNET
        from models.BinaryPlume3 import FlowMaskBinaryClassificationModel
        from models.EmissionRate2_4 import EmissionRateModel

        model = UNET(in_channels=hp.in_class, out_channels=hp.out_class).to(device)
        cont_model = ResNet(in_channels=hp.in_class + 1, out_channels=hp.out_class).to(device)
        BinaryModel = FlowMaskBinaryClassificationModel()
        EmissionRateModel = EmissionRateModel()

    elif hp.mode == "3d":
        pass

    model = torch.nn.DataParallel(model, device_ids=devicess)
    cont_model = torch.nn.DataParallel(cont_model, device_ids=devicess)
    BinaryModel = torch.nn.DataParallel(BinaryModel, device_ids=devicess)
    EmissionRateModel = torch.nn.DataParallel(EmissionRateModel, device_ids=devicess)

    print("load model:", hp.ckpt)
    print(os.path.join(hp.output_dir, hp.latest_checkpoint_file))

    ckpt = torch.load(
        os.path.join(hp.output_dir, "unet_" + hp.latest_checkpoint_file),
        map_location=lambda storage, loc: storage
    )

    cont_ckpt = torch.load(
        os.path.join(hp.output_dir, "cont_" + hp.latest_checkpoint_file),
        map_location=lambda storage, loc: storage
    )

    BinaryModel_ckpt = torch.load(
        os.path.join(hp.output_dir, "Binary_" + hp.latest_checkpoint_file),
        map_location=lambda storage, loc: storage
    )
    EmissionRateModel_ckpt = torch.load(
        os.path.join(hp.output_dir, "EmissionRate_" + hp.latest_checkpoint_file),
        map_location=lambda storage, loc: storage
    )

    # 加载模型的状态字典。状态字典包含了模型的所有参数
    model.load_state_dict(ckpt["model"])
    cont_model.load_state_dict(cont_ckpt["cont_model"])
    BinaryModel.load_state_dict(BinaryModel_ckpt["BinaryModel"])
    EmissionRateModel.load_state_dict(EmissionRateModel_ckpt["EmissionRateModel"])

    model.eval()
    cont_model.eval()
    BinaryModel.eval()
    EmissionRateModel.eval()

    test_dataset = XCh4Dataset(source_test_dir, label_test_dir, bin_label_test_dir, emission_label_test_dir)
    test_loader = DataLoader(test_dataset, batch_size=1)
    # 创建随机采样器
    # random_sampler = RandomSampler(test_dataset, replacement=False, num_samples=16)

    # 创建 DataLoader，使用随机采样器
    # test_loader = DataLoader(test_dataset, batch_size=16, sampler=random_sampler)

    # 存储评估指标以计算平均指标
    unet_acc_summary = []
    unet_jaccard_summary = []
    unet_f1_summary = []
    unet_pre_summary = []
    resnet_l1_summary = []  # 1范数

    bin_accuracy_summary = []
    bin_precision_summary = []
    bin_recall_summary = []
    bin_f1_summary = []

    emission_mape_summary = []
    emission_mae_summary = []

    # 存储所有预测结果和标签，以绘制散点图
    emission_predictions = []
    emission_labels = []

    #存储二分类模型的预测结果和标签，以计算二分类模型的评估指标
    # bin_all_predictions = []
    # bin_all_labels = []

    for subj in test_loader:
        # unet预测结果
        image = subj[0]  # 测试集图像

    # TODO:print(subj[0].dtype)
    for i, data_list in enumerate(test_loader):
        with torch.no_grad():  # 上下文管理器，用于禁用梯度计算。在测试阶段，模型不需要更新权重，因此不需要计算梯度
            #unet预测结果
            image = data_list[0].type(torch.FloatTensor)  # 测试集图像
            label = data_list[1]  # torch.Size([1, 80, 80])
            label = torch.unsqueeze(label, 1)  # 测试集标签 01   增加一个维度，使其形状变为[1, 1, 80, 80]
            output_tensor = model(image)
            # logits = torch.sigmoid(output_tensor)  # 对输出张量应用sigmoid函数，将其转换为概率值。
            # unet_predict = logits.clone()  # torch.Size([1, 1, 80, 80])
            unet_predict = output_tensor.clone()
            unet_predict = (unet_predict > 0.5).float()
            unet_predict[unet_predict >= 0.5] = 1
            unet_predict[unet_predict < 0.5] = 0
            dice, unet_acc, unet_precision, recall, unet_jaccard, unet_f1 = metrics_01(label.cpu(), unet_predict.cpu())

            # print("unet_acc:", unet_acc)
            # print("unet_precision:", unet_precision)

            # resnet预测结果
            cont_label = data_list[2]  #resnet标签
            cont_label = torch.unsqueeze(cont_label, 1)
            cont_predict = cont_model(
                torch.cat([image, unet_predict], dim=1))  # 将 input_tensor 和二值化后的 predict 张量在通道维度（dim=1）上进行拼接
            cont_predict = cont_predict.mul(unet_predict)  # 将 cont_predict 张量与二值化后的 predict 张量逐元素相乘
            resnet_l1 = metrics_cont(cont_label.cpu(), cont_predict.cpu())

            # Binary预测结果
            bin_label = data_list[3]  # 二分类标签
            image_for_bin = image.detach()
            unet_outputs_for_bin = unet_predict.detach()
            cont_outputs_for_bin = cont_predict.detach()
            bin_inputs = torch.cat([image_for_bin, unet_outputs_for_bin, cont_outputs_for_bin], dim=1)  # 拼接二分类模型输入的张量

            Binary_predict = BinaryModel(bin_inputs)  # 二分类模型输出
            Binary_predict = Binary_predict.clone()
            Binary_predict[Binary_predict > 0] = 1
            Binary_predict[Binary_predict <= 0] = 0

            bin_labels2 = bin_label.cpu().numpy()
            Binary_predict2 = Binary_predict.cpu().numpy()

            bin_accuracy = accuracy_score(bin_labels2, Binary_predict2)
            bin_precision = precision_score(bin_labels2, Binary_predict2)
            bin_recall = recall_score(bin_labels2, Binary_predict2, zero_division=1)
            bin_f1 = f1_score(bin_labels2, Binary_predict2)

            # EmissionRate预测结果
            label4 = data_list[4]  # emission标签
            image_for_emission = image.detach()
            unet_outputs_for_emission = unet_predict.detach()
            cont_outputs_for_emission = cont_predict.detach()
            bin_outputs_for_emission = Binary_predict.detach()
            bin_outputs_for_emission = bin_outputs_for_emission.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 80,
                                                                                                   80)  # 扩展维度
            emission_inputs = torch.cat(
                [image_for_emission, unet_outputs_for_emission, cont_outputs_for_emission, bin_outputs_for_emission],
                dim=1)

            EmissionRateOut = EmissionRateModel(emission_inputs)
            # print(EmissionRateOut)

            emission_predictions.extend(EmissionRateOut.cpu().numpy().flatten())
            emission_labels.extend(label4.cpu().numpy().flatten())

            # all_emissions = []
            # all_labels = []

            # all_emissions.extend(EmissionRateOut.cpu().numpy().flatten())
            # all_labels.extend(label4.cpu().numpy().flatten())

            # EmissionRateOut = EmissionRateOut.cpu().numpy().flatten()
            # label4 = label4.cpu().numpy().flatten()

            em_mae = mean_absolute_error(EmissionRateOut, label4)  # 计算平均绝对误差
            em_mape = mape_test(label4, EmissionRateOut)  # 计算平均百分比误差

        unet_jaccard_summary.append(unet_jaccard)
        unet_f1_summary.append(unet_f1)
        unet_acc_summary.append(unet_acc)
        unet_pre_summary.append(unet_precision)

        resnet_l1_summary.append(resnet_l1)

        bin_accuracy_summary.append(bin_accuracy)
        bin_precision_summary.append(bin_precision)
        bin_recall_summary.append(bin_recall)
        bin_f1_summary.append(bin_f1)

        emission_mape_summary.append(em_mape)
        emission_mae_summary.append(em_mae)

        name = test_dataset.image_files[i]  #文件名字

        """图片展示"""
        output_image = torch.squeeze(unet_predict) * 255  # torch.Size([80, 80]) # 预测
        output_image[output_image < 0] = 0
        output_image[output_image > 255] = 255
        gray_image = Image.fromarray(output_image.numpy().astype(np.uint8))

        label_out = torch.squeeze(label) * 255  # 标签
        label_out[label_out < 0] = 0
        label_out[label_out > 255] = 255
        label_image = Image.fromarray(label_out.numpy().astype(np.uint8))

        cont_label_out = torch.squeeze(cont_label) * 255  # 标签
        cont_label_out[cont_label_out < 0] = 0
        cont_label_out[cont_label_out > 255] = 255
        cont_label_image = Image.fromarray(cont_label_out.numpy().astype(np.uint8))

        cont_out = torch.squeeze(cont_predict) * 255
        cont_out[cont_out < 0] = 0
        cont_out[cont_out > 255] = 255
        cont_image = Image.fromarray(cont_out.numpy().astype(np.uint8))

        # 保存灰度图
        # label_image.save(os.path.join(output_dir_test, f"{name[:-5]}-label_int" + hp.save_arch))
        # cont_label_image.save(os.path.join(output_dir_test, f"{name[:-5]}-cont_label_int" + hp.save_arch))
        # gray_image.save(os.path.join(output_dir_test, f"{name[:-5]}-result_int" + hp.save_arch))
        # cont_image.save(os.path.join(output_dir_test, f"{name[:-5]}-cont_result_int" + hp.save_arch))

        # 保存彩色热力图
        # save_colored_heatmap(output_image, os.path.join(output_dir_test, f"{name[:-5]}-result_heatmap" + hp.save_arch))
        # save_colored_heatmap(label_out, os.path.join(output_dir_test, f"{name[:-5]}-label_heatmap" + hp.save_arch))
        # save_colored_heatmap(cont_label_out,
        #                      os.path.join(output_dir_test, f"{name[:-5]}-cont_label_heatmap" + hp.save_arch))

        # print(name)
        # print(EmissionRateOut[0][0])
        # print(np.float32(EmissionRateOut[0][0]))

        output_path = os.path.join(output_dir_test,
                                   f"{name[:-5]}" + f'_emisRate_{np.float32(EmissionRateOut[0][0]):.2f}_maxCon_{cont_predict.max():.4f}' + hp.save_arch)
        # print(os.path.join(output_dir_test, f"{name[:-5]}" +f'_emisRate_{np.float32(EmissionRateOut[0][0]):.2f}_maxCon_{cont_predict.max():.4f}' + hp.save_arch))

        save_colored_heatmap(cont_out, output_path)

    # unet的指标
    unet_acc_mean = np.mean(unet_acc_summary)
    unet_pre_mean = np.mean(unet_pre_summary)
    unet_jaccard_mean = np.mean(unet_jaccard_summary)
    unet_f1_mean = np.mean(unet_f1_summary)
    # resnet的指标
    resnet_l1_mean = np.mean(resnet_l1_summary)
    # bin的指标
    bin_acc_mean = np.mean(bin_accuracy_summary)
    bin_pre_mean = np.mean(bin_precision_summary)
    # bin_recall_mean = np.mean(bin_recall_summary)
    bin_f1_mean = np.mean(bin_f1_summary)
    # emission的指标
    emission_mape_mean = np.mean(emission_mape_summary)
    emission_L1loss_mean = np.mean(emission_mae_summary)
    # print("-----------------------------------------------")
    # print("emission_mape_summary:", emission_mape_summary)
    # print("Contains nan:", np.isnan(emission_mape_summary).any())
    # print("Mean:", np.mean(emission_mape_summary))
    # print("-----------------------------------------------")

    print(
        f"unet_acc_mean:{unet_acc_mean:.4f}, unet_pre_mean:{unet_pre_mean:.4f},unet_jaccard_mean:{unet_jaccard_mean:.4f}, unet_f1_mean:{unet_f1_mean:.4f}")
    print(f"resnet_l1_mean:{resnet_l1_mean:.4f}")
    print(f"bin_Acc_mean: {bin_acc_mean:.4f}, bin_Pre_mean: {bin_pre_mean:.4f}, bin_F1_mean: {bin_f1_mean:.4f}")
    print(f"emission_mape_mean: {emission_mape_mean:.4f}, emission_L1loss_mean: {emission_L1loss_mean:.4f}")

    data = {
        "pre": unet_precision,
        "rec": recall,
        "dice": dice,
        "jaccard": unet_jaccard,
        "f1": unet_f1,
    }

    index = sys.argv[0]
    # print(index)
    df = pd.DataFrame(
        data=data,
        columns=["pre", "rec", "dice", "jaccard", "f1"],
        index=[index]
    )
    df.to_csv(os.path.join("metrics_unetme.csv"), mode="a", header=False)  #

    true_emission_rates = np.random.uniform(0, 10000, 1000)
    predicted_emission_rates = true_emission_rates * np.random.normal(1, 0.1, 1000)

    plot_emission_rate(predicted_emission_rates, true_emission_rates)


from flask import Flask, jsonify,Response,send_from_directory
from flask_cors import CORS
app = Flask(__name__)
# 启用 CORS 支持
CORS(app)  # 默认允许所有来源的请求，具体配置可以更细致
import threading
import os
import  re
import  queue
import base64
data_queue=queue.Queue()
image_dir = "results/result3/"
pattern = re.compile(r"""
               (?P<letter>\w+)_(?P<id>\w+)_
               sza_(?P<sza>\d+\.\d+)_
               vza_(?P<vza>\d+\.\d+)_
               u10_(?P<u10>\d+\.\d+)_
               lon_(?P<lon>-?\d+\.\d+)_
               lat_(?P<lat>-?\d+\.\d+)_
               emisRate_(?P<emisRate>\d+\.\d+)_
               maxCon_(?P<maxCon>\d+\.\d+)
           """, re.VERBOSE)
def parse_image(dir):
    # total_files = 0  # 用于统计目录下的文件总数
    # parsed_files = 0  # 用于统计成功解析的文件数
    # queued_files = 0  # 用于统计成功放入队列的文件数
    # 解析图片目录，返回图片列表
    for file in os.listdir(dir):
        # total_files += 1  # 统计每个文件
        match = pattern.search(file)
        filepath=f"{image_dir}{file}"
        with open(filepath,"rb") as f:
            imgdata = base64.b64encode(f.read()).decode("utf-8")
        # parsed_files += 1  # 成功解析的文件数
        if match:
            info=match.groupdict()
            item= {"id": info["letter"]+"_"+info["id"], "filename": file, "info": info, "imgdata":imgdata}
            # print(item)
            if data_queue.full():
                print('队列已满')
                data_queue.get()
            data_queue.put(item)
            # queued_files+=1
        else:
            print(f"文件 {file} 无法解析")
        # 打印统计信息
        # print(f"总文件数: {total_files}")
        # print(f"成功解析的文件数: {parsed_files}")
        # print(f"成功放入队列的文件数: {queued_files}")
        # print(f"当前队列大小: {data_queue.qsize()}")

def ai_task():
    count=0
    while True:
        # test()
        count+=1
        parse_image(image_dir)

def start_data_thread():
    task1 = threading.Thread(target=ai_task)
    task1.daemon=True
    task1.start()



# @app.route('/img/<filename>', methods=['GET'])
# def get_img(filename):
#     return send_from_directory(image_dir, filename)


# 定义一个 GET 请求的接口
@app.route('/getdata', methods=['GET'])
def start():
    # def generate():
    #     while True:
    #         if not data_queue.empty():
    #             try:
    #                 parse_image(image_dir)
    #                 # 获取队列中的数据
    #                 data = data_queue.get_nowait()
    #                 # 返回数据并生成响应
    #                 yield json.dumps(data).encode('utf-8')
    #             except queue.Empty:
    #                 print("队列为空，结束生成器")
    #                 break  # 当队列为空时，退出循环，结束生成器
    #         else:
    #             print("队列为空，暂停")
    #             break
    data = []

    while not data_queue.empty():
        item = data_queue.get()
        data.append(item)
    # print(data)
    return jsonify(data)










if __name__ == "__main__":
    start_data_thread()
    app.run(debug=True,port=8087)
