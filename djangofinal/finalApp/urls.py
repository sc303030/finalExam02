from django.contrib import admin
from django.urls import path, include
from finalApp import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('index/', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('shop/', views.shop, name='shop'),
    path('selectshop/', views.selectshop, name='selectshop'),
    path('noonegu/<str:id>', views.noonegu, name='noonegu'),
    path('vegetableSelect/<str:id>', views.vegetableSelect, name='vegetableSelect'),
    path('mapseoulprice/', views.mapseoulprice, name='mapseoulprice'),
    path('mapseoulpriceajax/<str:id>', views.mapseoulpriceajax, name='mapseoulpriceajax'),
    path('seoulprice/', views.seoulprice, name='seoulprice'),
    path('news/', views.news, name='news'),
    path('cu/', views.cu, name='cu'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)