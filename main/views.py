from django.shortcuts import render
from drf_yasg import openapi
from drf_yasg.utils import swagger_auto_schema
from rest_framework.decorators import api_view
from rest_framework.generics import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from main.models import Area
from serializers import AreaSerializer


from swagger_interface import plumelist_request_body
# Create your views here.

@swagger_auto_schema(method='get', tags=['羽流中心点'],operation_summary="该接口用于获取地图上的羽流标点",
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
    # 这里可以根据实际需求生成数据，这里使用示例数据
    data = {
        "code": 200,
        "message": "操作成功",
        "data": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [40.70520933108145, 34.59370359949759]
                    },
                    "properties": {
                        "id": "CH4_6A_500m_-117.26766_34.59370",
                        "coordinates": [
                            [40.66520933108145, 35.01431581430332],
                            [40.66520933108145, 35.070895256143764],
                            [40.73431608304051, 35.070895256143764],
                            [40.73431608304051, 35.01431581430332]
                        ]
                    }
                }
            ]
        }
    }
    return Response(data, status=status.HTTP_200_OK)
@swagger_auto_schema(method='get', tags=['羽流详细信息'],operation_summary="获取羽流详情信息",
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

@swagger_auto_schema(method='get', tags=['羽流图片'],operation_summary="根据ID获取图片",
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
    area= get_object_or_404(Area, id=id)
    if area:
        data = AreaSerializer(area).data.get("methane")["concentration_image"]
        return Response(data, status=status.HTTP_200_OK)
    return Response(status=status.HTTP_404_NOT_FOUND)


