from drf_yasg import openapi

plumelist_request_body = openapi.Schema(
    type=openapi.TYPE_OBJECT,
    required=["None"],
    properties={

    },
    example={
        'None': '没有参数',
    }
)