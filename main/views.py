import json

from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.generics import get_object_or_404
from rest_framework.response import Response

from main.models import Area, Methane
from serializers import AreaSerializer
import base64
# Create your views here.
url = "http://127.0.0.1:8087/getdata"


def get_data():
    print("正在获取数据")
    import requests
    while True:
        try:
            response = requests.get(url)
            # response.raise_for_status()
            datas = response.json()
            for data in datas:
                # 创建或更新 Methane 数据
                imgbase64 = data["imgdata"].encode('utf-8')
                # print(imgbase64)
                imgdata = base64.b64decode(imgbase64)
                # 保存到指定文件下
                filepath = "media/Methane/" + data['filename']
                with open(filepath, "wb") as f:
                    f.write(imgdata)
                # print("保存成功")


                methane_data, created = Methane.objects.update_or_create(
                    name=data['id'],
                    defaults={
                        'concentration_image': data['filename'],  # 假设你有文件名的路径
                        'emission_rate': data['info']['emisRate'],
                        'max_concentration': data['info']['maxCon'],
                    }
                )
                # 创建或更新 Area 数据
                area_data, created = Area.objects.update_or_create(
                    name=data['id'],  # 假设区域的名称就是ID
                    defaults={
                        'longitude': data['info']['lon'],
                        'latitude': data['info']['lat'],
                        'sza': data['info']['sza'],
                        'vza': data['info']['vza'],
                        'u10': data['info']['u10'],
                        'methane': methane_data  # 关联 Methane 模型
                    }
                    )
        except Exception as e:
            print("获取数据失败:",e)



@swagger_auto_schema(method='get', tags=['羽流中心点'], operation_summary="该接口用于获取地图上的羽流标点",
                     responses={200: '返回成功'},
                     manual_parameters=[
                         openapi.Parameter(
                             'id',  # 参数名
                             openapi.IN_QUERY,  # 参数位置，这里是查询参数
                             description="羽流的 ID",  # 参数描述
                             type=openapi.TYPE_STRING  # 参数类型
                         )
                     ],
                     )
@api_view(['GET'])
def PlumeListAPI(request):
    areas = Area.objects.all()
    features = []
    for area in areas:
        methane = area.methane
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",  # 这里假设是点数据，如果是其他几何类型，可以修改
                "coordinates": [area.longitude, area.latitude]  # 使用从数据库中提取的经纬度
            },
            "properties": {
                "id": f"CH4_{area.id}_500m_{area.longitude}_{area.latitude}",
                "coordinates": [
                    # 这里假设有一个坐标数组，用来表示一个矩形/多边形等
                    [area.longitude - 0.01, area.latitude - 0.01],
                    [area.longitude - 0.01, area.latitude + 0.01],
                    [area.longitude + 0.01, area.latitude + 0.01],
                    [area.longitude + 0.01, area.latitude - 0.01]
                ],
                "sza": area.sza,  # 太阳天顶角
                "vza": area.vza,  # 视天顶角
                "u10": area.u10,  # 风速
                "emisRate": methane.emission_rate if methane else None,  # 甲烷排放率
                "maxCon": methane.max_concentration if methane else None,  # 甲烷最大浓度
                "region_name": area.name,  # 区域名称
            }
        }
        features.append(feature)
        # 构造完整的响应数据
    data = {
        "code": 200,
        "message": "操作成功",
        "data": {
            "type": "FeatureCollection",
            "features": features
        }
    }
    return Response(data, status=status.HTTP_200_OK)


@swagger_auto_schema(method='get', tags=['羽流详细信息'], operation_summary="获取羽流详情信息",
                     responses={200: '返回成功'},
                     manual_parameters=[
                         openapi.Parameter(
                             'id',  # 参数名
                             openapi.IN_QUERY,  # 参数位置，这里是查询参数
                             description="羽流的 ID",  # 参数描述
                             type=openapi.TYPE_STRING  # 参数类型
                         )
                     ],
                     )
@api_view(['GET'])
def PlumeInfoAPI(request):
    id = request.GET.get('id')
    area = get_object_or_404(Area, id=id)
    if area:
        data = AreaSerializer(area).data
        return Response(data, status=status.HTTP_200_OK)

    return Response(status=status.HTTP_404_NOT_FOUND)
    # data = {
    # "code": 200,
    # "message": "操作成功",
    # "data": {
    # }


@swagger_auto_schema(method='get', tags=['羽流图片'], operation_summary="根据ID获取图片",
                     responses={200: '返回成功'},
                     manual_parameters=[
                         openapi.Parameter(
                             'id',  # 参数名
                             openapi.IN_QUERY,  # 参数位置，这里是查询参数
                             description="羽流的ID",  # 参数描述
                             type=openapi.TYPE_STRING  # 参数类型
                         )
                     ],
                     )
@api_view(['GET'])
def PlumImageAPI(request):
    id = request.GET.get('id')
    area = get_object_or_404(Area, id=id)
    if area:
        data = AreaSerializer(area).data.get("methane")["concentration_image"]
        return Response(data, status=status.HTTP_200_OK)
    return Response(status=status.HTTP_404_NOT_FOUND)
