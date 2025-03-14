# main/apps.py
from django.apps import AppConfig


class MainConfig(AppConfig):
    name = 'main'

    def ready(self):
        from django.db.models.signals import post_migrate
        from django.core.management import call_command
        import threading
        print("服务启动完成")
        # 在这里启动线程
        def getdata_threading():
            task_data = threading.Thread(target=self.start_get_data_task)
            task_data.daemon = True
            task_data.start()

        # 调用线程任务
        getdata_threading()

    def start_get_data_task(self):
        from main.views import get_data
        get_data()
