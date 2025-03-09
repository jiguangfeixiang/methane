from main.models import *
from django.contrib import admin
admin.site.site_title = '甲烷管理'
admin.site.site_header = '甲烷后台'
admin.site.index_title = '甲烷管理'


# 管理 Methane 模型
@admin.register(Methane)
class MethaneAdmin(admin.ModelAdmin):
    # 在列表中显示的字段
    list_display = ['id', 'concentration_image', 'emission_rate', 'max_concentration']

    # 设置搜索字段
    search_fields = ['emission_rate', 'max_concentration']  # 可搜索的字段

    # 设置筛选字段
    list_filter = ['emission_rate', 'max_concentration']  # 可筛选的字段


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


