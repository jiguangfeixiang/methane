# main/apps.py
from django.apps import AppConfig


class MainConfig(AppConfig):
    name = 'main'
    verbose_name = '甲烷数据管理'

    def ready(self):
        from django.db.models.signals import post_migrate
        from django.core.management import call_command
        import threading
        print("服务启动完成")

        # import tifffile
        #
        # try:
        #     img = tifffile.imread("D:/onedrive/图片/A_0Q600_sza_19.28_vza_9.27_u10_2.69_lon_6.15_lat_31.83.tiff")
        #     print(f"成功使用 tifffile 读取 TIFF 图像，图像形状：{img.shape}")
        # except Exception as e:
        #     print(f"读取 TIFF 文件失败：{e}")

        # 在这里启动线程
        # def getdata_threading():
        #     task_data = threading.Thread(target=self.start_get_data_task)
        #     task_data.daemon = True
        #     task_data.start()

        # 调用线程任务
        # getdata_threading()

    # def start_get_data_task(self):
    #     from main.views import get_data
    #     get_data()
