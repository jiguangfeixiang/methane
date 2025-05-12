from drf_yasg import openapi
from drf_yasg.views import get_schema_view
from rest_framework import permissions
from django.conf import  settings
from main.views import PlumeListAPI, PlumeInfoAPI, PlumImageAPI, get_data, GetWsToken

schema_view = get_schema_view(
    openapi.Info(
        title="甲烷接口文档",
        description="文档描述",
        default_version='v1',
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@sample.local"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static

urlpatterns = [
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path('admin/', admin.site.urls),
    path('main/plumeinfo/', PlumeInfoAPI, name='甲烷信息获取'),

    path('main/plume/', PlumImageAPI, name='甲烷图片获取'),

    path('main/plumelist/',PlumeListAPI,name='甲烷坐标获取'),
    # path('main/getdata/',get_data,name='数据获取'),
    path('getchatboxtoken/', GetWsToken,name='GetWsToken获取'),
    
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL,
                                                                                           document_root=settings.MEDIA_ROOT)  # 媒体文件路径
