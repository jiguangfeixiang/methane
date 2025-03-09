from rest_framework import serializers

from main.models import Methane, Area


class MethaneSerializer(serializers.ModelSerializer):
    class Meta:
        model = Methane
        fields = '__all__'
class AreaSerializer(serializers.ModelSerializer):
    methane = MethaneSerializer(many=False)
    class Meta:
        model = Area
        fields = '__all__'