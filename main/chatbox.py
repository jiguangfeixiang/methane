from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.lke.v20231130 import lke_client, models
import json
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

models_appkey = {
    "Deepseek R1": "xizHZOiO",
    "Deepseek V3": "BBptYfKZ",
    "Deepseek V3-0324": "FAKdpKcu",
}

config = {
    'secretId': 'AKIDgDWbS5TilaAWdWByYvzpQnvpVw97sY0D',
    'secretKey': 'KSN3VjiW6SZR5VnqfFbkeKn22pxsYzBD',
    'appId': 'FAKdpKcu'
}
@api_view(['GET'])
def GetWsToken(request):
    model_type = request.query_params.get('model_type')
    for key, value in models_appkey.items():
        if model_type == key:
            config['appId'] = value
            break
    try:
        # 创建凭证对象
        cred = credential.Credential(config['secretId'], config['secretKey'])
        
        # 配置HTTP Profile
        http_profile = HttpProfile()
        http_profile.endpoint = "lke.ap-guangzhou.tencentcloudapi.com"
        
        # 配置Client Profile
        client_profile = ClientProfile()
        client_profile.httpProfile = http_profile

        # 创建客户端
        client = lke_client.LkeClient(cred, "ap-guangzhou", client_profile)
        
        # 创建请求对象
        req = models.GetWsTokenRequest()
        req.Type = 5
        req.BotAppKey = config['appId']
        req.VisitorBizId = config['appId']
        
        # 发送请求
        resp = client.GetWsToken(req)
        
        # 返回响应
        result = {
            'code': 200,
            'data': {
              'apiResponse': json.loads(resp.to_json_string()),
            },
            'msg': '成功获取WsToken',
        }
        # return Response(result,  status=status.HTTP_200_OK)
        return Response(result,  status=status.HTTP_200_OK, headers={'Access-Control-Allow-Origin': '*'})
    except TencentCloudSDKException as e:
        print(e)
        result = {
            'code': 500,
            'data': {
                'error': str(e)
            },
            'msg': '获取WsToken失败',
        }
        return Response(result,  status=status.HTTP_500_INTERNAL_SERVER_ERROR)