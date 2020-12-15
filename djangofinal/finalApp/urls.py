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
    path('news/', views.news, name='news'),
    path('distribution/', views.distribution, name='distribution'),
    path('cart/', views.cart, name='cart'),
    path('news/', views.news, name='news'),
    path('singlenews/', views.singlenews, name='singlenews'),
    path('page404/', views.page404, name='page404'),
    path('checkout/', views.checkout, name='checkout'),
    path('vegetableSelectProducer/<str:id>', views.vegetableSelectProducer, name='vegetableSelectProducer'),
    path('bigdatatell/', views.bigdatatell, name='bigdatatell'),
    path('mapkakao/', views.mapkakao, name='mapkakao'),
    path('get_map_kakao/<str:id>', views.get_map_kakao, name='get_map_kakao'),
    path('additionalfactors/', views.additionalfactors, name='additionalfactors'),
    path('additionalfactors2/', views.additionalfactors2, name='additionalfactors2'),
    path('search/', views.search, name='search'),
    path('predict/', views.predict, name='predict'),
    path('ttt/', views.ttt, name='ttt'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)