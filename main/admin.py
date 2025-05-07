import base64
import json
import os
import re

from django.core.files.base import ContentFile
from django.utils.html import format_html
from django.utils.safestring import mark_safe

from main.models import *
from django.contrib import admin
import requests
from django.core.files import File
from django.conf import settings

admin.site.site_title = '甲烷管理'
admin.site.site_header = '甲烷后台'
admin.site.index_title = '甲烷管理'

img_url = "http://192.168.241.79:8000/methaneImg"
write_dir = os.path.join(settings.MEDIA_ROOT, 'MethaneShowImg')
#甲烷热力图
# 管理 Methane 模型
@admin.register(Methane)
class MethaneAdmin(admin.ModelAdmin):
    # 在列表中显示的字段
    list_display = ['id', 'concentration_image_preview', 'emission_rate', 'max_concentration']

    # 设置搜索字段
    search_fields = ['emission_rate', 'max_concentration']  # 可搜索的字段

    # 设置筛选字段
    list_filter = ['emission_rate', 'max_concentration']  # 可筛选的字段

    # 定义自定义的方法来显示图片
    def concentration_image_preview(self, obj):
        if obj.concentration_image:
            return mark_safe(f'<img src="{obj.concentration_image.url}" width="100" />')
        return "No image"

    concentration_image_preview.short_description = '甲烷热力图'  # 设置列名

#区域甲烷信息查看
# 管理 Area 模型
@admin.register(Area)
class AreaAdmin(admin.ModelAdmin):
    # 在列表中显示的字段
    list_display = ['id', 'name', 'longitude', 'latitude', 'sza', 'vza', 'u10', 'methane']

    # 设置搜索字段
    search_fields = ['name', 'longitude', 'latitude']  # 可搜索的字段

    # 设置筛选字段
    list_filter = ['sza', 'vza', 'u10', 'methane']  # 可筛选的字段

    # 设置排序规则
    ordering = ['name']  # 默认按名称排序





#甲烷上传文件
@admin.register(MethaneUploadImg)
class MethaneUploadImg(admin.ModelAdmin):
    import tifffile
    list_display = ['id',"upload_image","show_result_image", 'show_info']

    def save_model(self, request, obj, form, change):
        super().save_model(request, obj, form, change)
        image_path = obj.upload_image.path
        # 打印图片的数据
        imgdata = obj.upload_image.read()
        # 保存图片到指定目录
        if obj.upload_image:
            # 1. 准备图片路径
            print(obj.upload_image.name)
            filename = os.path.basename(obj.upload_image.name)
            print(filename)
            try:
                # 2. 发送请求到 AI 推理端
                with open(image_path, 'rb') as f:
                    response = requests.post(
                        img_url,  # 假设你的AI服务在这个地址
                        files={'image': f},
                        data={'filename': filename},  # 字符串放 data
                        timeout=5
                    )
                res = response.json()
                print("传过来的数据：",res)
            except:
                print("请求失败")
                return obj
            # 假设 'res' 包含 'info' 字段并且是一个字典
            data = {
                "img":res["heatmap"],
                "con": res["con"],
                "emis": float(res['emis']),  # 将 emis 转换为 float 类型
                "sza": res['SZA'],
                "vza": float(res['VZA']),  # 将 vza 转换为 float 类型
                "u10": float(res['U10'])  # 将 u10 转换为 float 类型
            }
            # print(data)
            # 确保 obj.info 是一个字典，如果它是 None，则初始化为空字典
            if obj.info is None:
                obj.info = {}
            # 把图片写进去show_image
            obj.info["con"]=data["con"]
            obj.info["emis"]=data["emis"]
            obj.info["SZA"]=data["sza"]
            obj.info["VZA"]=data["vza"]
            obj.info["U10"]=data["u10"]
            imgdata = base64.b64decode(data["img"])
            # 给当前对象的show_img赋值
            obj.show_image.save(filename, ContentFile(imgdata), save=True)
            obj.save()
            return obj

    def show_result_image(self, obj):
        if obj.show_image:
            return format_html('<img src="{}" width="200"/>', obj.show_image.url)
        return "无"

    def show_send_image(self, obj):
        if obj.upload_image:
            return format_html('<img src="{}" width="200"/>', obj.upload_image.url)
        return "无"

    def show_info(self, obj):
        if hasattr(obj, 'info') and obj.info:
            # 格式化展示字典中的信息
            info_html = "<br>".join([f"{key}: {value}" for key, value in obj.info.items()])
            return format_html(f"<div>{info_html}</div>")
        return "无信息"

    show_result_image.short_description = "AI返回图"
    show_send_image.short_description = "上传图"
    show_info.short_description = "AI返回的信息"
