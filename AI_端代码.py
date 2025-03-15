
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
            item= {"id": info["id"], "filename": file, "info": info, "imgdata":imgdata}
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
