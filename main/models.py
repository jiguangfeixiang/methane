from django.db import models


class Methane(models.Model):
    # 甲烷浓度热力图
    concentration_image = models.ImageField(upload_to='Methane/', verbose_name='甲烷浓度热力图',
                                            blank=True, null=True)

    # 甲烷排放率
    emission_rate = models.FloatField(verbose_name='甲烷排放率', null=True, blank=True)

    # 甲烷最高浓度值
    max_concentration = models.FloatField(verbose_name='甲烷最高浓度值', null=True, blank=True)


# Create your models here.
class Area(models.Model):
    # 区域名称
    name = models.CharField(max_length=100, verbose_name='区域名称')
    # 经纬度
    longitude = models.FloatField(verbose_name='经度', null=True, blank=True)
    latitude = models.FloatField(verbose_name='纬度', null=True, blank=True)

    # 太阳天顶角 (SZA)
    sza = models.FloatField(verbose_name='太阳天顶角(SZA)', null=True, blank=True)

    # 视天顶角 (VZA)
    vza = models.FloatField(verbose_name='视天顶角(VZA)', null=True, blank=True)

    # 10米高度的风速
    u10 = models.FloatField(verbose_name='10米高度的风速(U10)', null=True, blank=True)

    # 建立一个外键，一个区域对应一片甲烷
    methane = models.ForeignKey(Methane, on_delete=models.CASCADE, verbose_name='甲烷', null=True, blank=True)
    def __str__(self):
        return self.name