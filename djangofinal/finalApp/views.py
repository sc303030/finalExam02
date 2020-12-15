from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from .models import *
from django.views.decorators.csrf import csrf_exempt
import csv
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor

from sklearn.exceptions import NotFittedError

from xgboost import XGBRegressor


import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.ensemble import VotingRegressor

from xgboost import plot_importance

from sklearn.datasets import load_diabetes
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor

from datetime import datetime, timedelta

# Create your views here.

def index(request):
    return render(request, 'finalApp/index_2.html')


def ttt(request):
    return render(request, 'finalApp/about_3.html')

def about(request):
    return render(request, 'finalApp/about_2.html')


def shop(request):
    return render(request, 'finalApp/shop_3.html')


def selectshop(request):
    return render(request, 'finalApp/selectshop.html')

def news(request):
    return render(request, 'finalApp/news.html')


def mapseoulprice(request):
    return render(request, 'finalApp/seoul_map_price.html')

def distribution(request):
    return render(request, 'finalApp/distribution.html')

def cart(request):
    return render(request, 'finalApp/cart.html')

def news(request):
    return render(request, 'finalApp/news.html')

def singlenews(request):
    return render(request, 'finalApp/single-news.html')

def page404(request):
    return render(request, 'finalApp/404.html')
def checkout(request):
    return render(request, 'finalApp/checkout.html')


def bigdatatell(request):
    return render(request, 'finalApp/bigdatatell.html')

def mapkakao(request):
    return render(request, 'finalApp/map_kakao.html')

def additionalfactors(request):
    return render(request, 'finalApp/additionalfactors.html')



def noonegu(request, id):
    price_mart = []
    price_si = []
    price_seoul = []
    category = []
    name = id
    dada = []
    with open('./static/seoul_mart_jang_graph.csv', mode='r', encoding='utf-8') as seoul_lists:
        reader = csv.reader(seoul_lists)

        for list_num in reader:
            if list_num[4] == str(id):
                if list_num[1] == '':
                    price_si.append(0)
                    price_mart.append(int(round(float(list_num[2]), 0)))
                    price_seoul.append(int(round(float(list_num[3]), 0)))
                    category.append(list_num[5])
                    dada.append(list_num[0])


                elif list_num[2] == '':
                    price_si.append(int(round(float(list_num[1]),0)))
                    price_mart.append(0)
                    price_seoul.append(int(round(float(list_num[3]),0)))
                    category.append(list_num[5])
                    dada.append(list_num[0])

                elif list_num[3] == '':
                    price_si.append(int(round(float(list_num[1]),0)))
                    price_mart.append(int(round(float(list_num[2]),0)))
                    price_seoul.append(0)
                    category.append(list_num[5])
                    dada.append(list_num[0])

                elif list_num[3] != '' and list_num[2] != '' and list_num[1] != '':
                    price_si.append(int(round(float(list_num[1]),0)))
                    price_mart.append(int(round(float(list_num[2]),0)))
                    price_seoul.append(int(round(float(list_num[3]),0)))
                    category.append(list_num[5])
                    dada.append(list_num[0])

    name_list = [name for _ in range(len(category))]
    csvData = [[price_si[i], price_mart[i], price_seoul[i], dada[i], category[i], name_list[i]] for i in
               range(len(category))]

    with open('./static/seoul_mart_jang_graph_select.csv',  mode='w', encoding='utf-8-sig', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        for i in csvData:
            wr.writerow(i)

    cheaper_price_si = []
    cheaper_price_mart = []

    label_mart = []
    label_si = []

    expensive_li = []
    cheap_li = []

    with open('./static/jang_mart_price_vs.csv', mode='r', encoding='utf-8-sig') as seoul_lists02:
        reader = csv.reader(seoul_lists02)

        for list_num02 in reader:
            if list_num02[0] == str(id) and list_num02[3] == '시장':
                cheaper_price_mart.append(str(list_num02[4]))
            elif list_num02[0] == str(id) and list_num02[3] == '마트':
                cheaper_price_si.append(str(list_num02[4]))

        for i in cheaper_price_mart:
            if i == '무':
                # c_m_image_src_li.append("Radish.jpg")
                label_mart.append(1)
            elif i == '배추':
                label_mart.append(2)
            elif i == '상추':
                label_mart.append(3)
            elif i == '양파':
                label_mart.append(4)
            elif i == '오이':
                label_mart.append(5)

        for i in cheaper_price_si:
            if i == '무':
                label_si.append(1)
            elif i == '배추':
                label_si.append(2)
            elif i == '상추':
                label_si.append(3)
            elif i == '양파':
                label_si.append(4)
            elif i == '오이':
                label_si.append(5)
    print(label_mart)
    print('*' * 50)
    print(label_si)
    with open('./static/dict_expensive_cheap.csv', mode='r') as dict_expensive_cheap:
        reader = csv.reader(dict_expensive_cheap)

        for ex_li in reader:
            if ex_li[0] == str(id):
                expensive_li.append(str(ex_li[1]))
                cheap_li.append(str(ex_li[2]))

        expensive_li = ','.join(expensive_li)
        cheap_li = ','.join(cheap_li)

    local = []
    category_local = []
    ratio = []
    expen_cheap = []

    with open('./static/gu_expen_cheaper.csv', mode='r', encoding='utf-8-sig') as seoul_lists:
        reader = csv.reader(seoul_lists)

        for list_num in reader:
            if list_num[0] == str(id):
                local.append(list_num[0])
                category_local.append(list_num[1])
                ratio.append(list_num[2])
                expen_cheap.append(list_num[3])

    csvData4 = [[local[i], category_local[i], ratio[i], expen_cheap[i]] for i in
               range(len(local))]

    with open('./static/gu_expen_cheaper_select.csv', mode='w', encoding='utf-8-sig', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        for i in csvData4:
            wr.writerow(i)
            print(i)
    local_next = []
    cheaper_next = []
    expen_next = []
    category_next = []



    with open('./static/local_expen_cheaper.csv', mode='r', encoding='utf-8') as seoul_lists:
        reader = csv.reader(seoul_lists)

        for list_num in reader:
            if list_num[0] == str(id):
                if list_num[1] == 'nothing':
                    local_next.append(list_num[0])
                    cheaper_next.append('無')
                    expen_next.append(list_num[2])
                    category_next.append(list_num[3])
                elif list_num[2] == 'nothing':
                    local_next.append(list_num[0])
                    cheaper_next.append(list_num[1])
                    expen_next.append('無')
                    category_next.append(list_num[3])
                elif list_num[1] != 'nothing' and list_num[2] != 'nothing':
                    local_next.append(list_num[0])
                    cheaper_next.append(list_num[1])
                    expen_next.append(list_num[2])
                    category_next.append(list_num[3])

    csvData5 = [[local_next[i], cheaper_next[i], expen_next[i], category_next[i]] for i in
                range(len(category_next))]

    with open('./static/local_expen_cheaper_select.csv', mode='w', encoding='utf-8-sig', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        for i in csvData5:
            wr.writerow(i)

    year2020_mean = []
    year2020_local = []
    year2020_place = []
    year2020_category = []

    with open('./static/seoul_1year_mean.csv', mode='r', encoding='utf-8-sig') as seoul_lists:
        reader = csv.reader(seoul_lists)

        for list_num in reader:
            if list_num[1] == str(id):
                a = int(round(float(list_num[0]),0))
                number = format(a, ',')
                year2020_mean.append(number)
                year2020_local.append(list_num[1])
                year2020_place.append(list_num[2])
                year2020_category.append(list_num[3])

    csvData6 = [[year2020_mean[i], year2020_local[i], year2020_place[i], year2020_category[i]] for i in
                range(len(year2020_category))]

    with open('./static/seoul_1year_mean_select.csv', mode='w', encoding='utf-8-sig', newline='') as csvfile:
        wr = csv.writer(csvfile, delimiter=',')
        for i in csvData6:
            wr.writerow(i)


    context = {
        'price_si': price_si,
        'price_mart': price_mart,
        'price_seoul': price_seoul,
        'name': name,
        'dada': dada,
        'category': category[0],

        'expensive_li': expensive_li,
        'cheap_li': cheap_li,

        'label_mart': label_mart,
        'label_si': label_si,

        'cheaper_price_si': cheaper_price_si,
        'cheaper_price_mart': cheaper_price_mart,
    }
    return render(request, 'finalApp/noonegu.html', context)

@csrf_exempt
def vegetableSelect(request, id):
    print(id)
    print('----------- ajax json vegetableSelect')
    price_si = []
    price_mart = []
    price_seoul = []
    category = []
    name = []
    dada = []
    with open('./static/seoul_mart_jang_graph_select.csv', mode='r', encoding='utf-8-sig') as vegetable_lists:
        reader = csv.reader(vegetable_lists)

        for list_num in reader:
            if list_num[4] == str(id):
                if list_num[0] == '':
                    price_si.append(0)
                    price_mart.append(int(list_num[1]))
                    price_seoul.append(int(list_num[2]))
                    dada.append(list_num[3])
                    category.append(list_num[4])
                    name.append(list_num[5])


                elif list_num[1] == '':
                    price_si.append(int(list_num[0]))
                    price_mart.append(0)
                    price_seoul.append(int(list_num[2]))
                    dada.append(list_num[3])
                    category.append(list_num[4])
                    name.append(list_num[5])

                elif list_num[2] == '':
                    price_si.append(int(list_num[0]))
                    price_mart.append(int(list_num[1]))
                    price_seoul.append(0)
                    dada.append(list_num[3])
                    category.append(list_num[4])
                    name.append(list_num[5])

                elif list_num[2] != '' and list_num[1] != '' and list_num[0] != '':
                    price_si.append(int(list_num[0]))
                    price_mart.append(int(list_num[1]))
                    price_seoul.append(int(list_num[2]))
                    dada.append(list_num[3])
                    category.append(list_num[4])
                    name.append(list_num[5])

    local = []
    category_local = []
    ratio = []
    ExpCheap = []


    with open('./static/gu_expen_cheaper_select.csv', mode='r', encoding='utf-8-sig') as vegetable_lists:
        reader = csv.reader(vegetable_lists)

        for list_num in reader:
            if list_num[1] == str(id):
                local.append(list_num[0])
                category_local.append(list_num[1])
                ratio.append(list_num[2])
                if str(list_num[3]) == '비싸다':
                    ExpCheap.append(str('비싸게'))
                else:
                    ExpCheap.append(str('싸게'))

    local_next = []
    cheaper_next = []
    expen_next = []
    category_next = []

    with open('./static/local_expen_cheaper_select.csv', mode='r', encoding='utf-8-sig') as vegetable_lists:
        reader = csv.reader(vegetable_lists)

        for list_num in reader:
            if list_num[3] == str(id):
                local_next.append(list_num[0])
                cheaper_next.append(list_num[1])
                expen_next.append(list_num[2])
                category_next.append(list_num[3])

    year2020_mean = []
    year2020_place = []

    with open('./static/seoul_1year_mean_select.csv', mode='r', encoding='utf-8-sig') as vegetable_lists:
        reader = csv.reader(vegetable_lists)

        for list_num in reader:
            if list_num[3] == str(id):
                year2020_mean.append(list_num[0])
                year2020_place.append(list_num[2])

    dada2 = []

    for i in range(1,len(dada)+1):
        if i % 3 == 0:
            dada2.append(dada[i-1][:4] + '년' + dada[i-1][5:7] + '월')
        else:
            dada2.append(' ')

    data_price_change = []
    onion_change = []
    cabbage_change = []
    raddish_change = []
    cucumber_change = []
    lettuce_change = []

    with open('./static/seoul_price_change.csv', mode='r', encoding='utf-8-sig') as vegetable_lists:
        reader = csv.reader(vegetable_lists)

        for list_num in reader:
            if str(id) == '양파':
                data_price_change.append(list_num[0][0:4] + '년' + list_num[0][5:7] + '월')
                onion_change.append(round(float(list_num[1]),2))
            elif str(id) == '배추':
                data_price_change.append(list_num[0][0:4] + '년' + list_num[0][5:7] + '월')
                cabbage_change.append(round(float(list_num[2]),2))
            elif str(id) == '무':
                data_price_change.append(list_num[0][0:4] + '년' + list_num[0][5:7] + '월')
                raddish_change.append(round(float(list_num[3]),2))
            elif str(id) == '오이':
                data_price_change.append(list_num[0][0:4] + '년' + list_num[0][5:7] + '월')
                cucumber_change.append(round(float(list_num[4]),2))
            else:
                data_price_change.append(list_num[0][0:4] + '년' + list_num[0][5:7] + '월')
                lettuce_change.append(round(float(list_num[5]),2))

    onion_year = []
    onion_plmi = []
    onion_ratio = []
    onion_infor = []

    cabbage_year = []
    cabbage_plmi = []
    cabbage_ratio = []
    cabbage_infor = []

    radish_year = []
    radish_plmi = []
    radish_ratio = []
    radish_infor = []

    cucumber_year = []
    cucumber_plmi = []
    cucumber_ratio = []
    cucumber_infor = []

    lettuce_year = []
    lettuce_plmi = []
    lettuce_ratio = []
    lettuce_infor = []

    onion_length = []
    cabbage_length = []
    radish_length = []
    cucumber_length = []
    lettuce_length = []


    with open('./static/vegi_change.csv', mode='r', encoding='utf-8-sig') as bigdatatell2_list:
        reader = csv.reader(bigdatatell2_list)

        for list_num in reader:
            if  list_num[0] == '양파':
                onion_length.append(list_num[0])
                onion_year.append(list_num[1])
                onion_plmi.append(list_num[2])
                onion_ratio.append(list_num[3])
                onion_infor.append(list_num[4])
            elif list_num[0] == '배추':
                cabbage_length.append(list_num[0])
                cabbage_year.append(list_num[1])
                cabbage_plmi.append(list_num[2])
                cabbage_ratio.append(list_num[3])
                cabbage_infor.append(list_num[4])
            elif list_num[0] == '무':
                radish_length.append(list_num[0])
                radish_year.append(list_num[1])
                radish_plmi.append(list_num[2])
                radish_ratio.append(list_num[3])
                radish_infor.append(list_num[4])
            elif list_num[0] == '오이':
                cucumber_length.append(list_num[0])
                cucumber_year.append(list_num[1])
                cucumber_plmi.append(list_num[2])
                cucumber_ratio.append(list_num[3])
                cucumber_infor.append(list_num[4])
            else:
                lettuce_length.append(list_num[0])
                lettuce_year.append(list_num[1])
                lettuce_plmi.append(list_num[2])
                lettuce_ratio.append(list_num[3])
                lettuce_infor.append(list_num[4])

    context = {
        'price_si': price_si,
        'price_mart': price_mart,
        'price_seoul': price_seoul,
        'name': name[0],
        'category': category[0],
        "dada":dada2,

        'ratio': ratio,
        'ExpCheap': ExpCheap,

        'expen_next': expen_next,
        'cheaper_next': cheaper_next,

        'year2020_mean': year2020_mean,
        'year2020_place': year2020_place,
        'length': len(year2020_place),

        'data_price_change': data_price_change,
        'onion_change': onion_change,
        'cabbage_change': cabbage_change,
        'raddish_change': raddish_change,
        'cucumber_change': cucumber_change,
        'lettuce_change': lettuce_change,
        'daylength': len(data_price_change),

        'onion_year': onion_year,
        'onion_plmi': onion_plmi,
        'onion_ratio': onion_ratio,
        'onion_infor': onion_infor,

        'cabbage_year': cabbage_year,
        'cabbage_plmi': cabbage_plmi,
        'cabbage_ratio': cabbage_ratio,
        'cabbage_infor': cabbage_infor,

        'radish_year': radish_year,
        'radish_plmi': radish_plmi,
        'radish_ratio': radish_ratio,
        'radish_infor': radish_infor,

        'cucumber_year': cucumber_year,
        'cucumber_plmi': cucumber_plmi,
        'cucumber_ratio': cucumber_ratio,
        'cucumber_infor': cucumber_infor,

        'lettuce_year': lettuce_year,
        'lettuce_plmi': lettuce_plmi,
        'lettuce_ratio': lettuce_ratio,
        'lettuce_infor': lettuce_infor,

        'onion_length' : len(onion_length),
        'cabbage_length': len(cabbage_length),
        'radish_length': len(radish_length),
        'cucumber_length': len(cucumber_length),
        'lettuce_length': len(lettuce_length)
    }
    if price_si[0] == 0:
        context['sizero'] = 'False'

    if price_mart[0] == 0:
        context['martzero'] = 'False'
    if price_seoul[0] == 0:
        context['seoulzero'] = 'False'

    if price_si[0] == 0 and price_mart[0] == 0 and price_seoul[0] == 0:
        context['priceZero'] = 'False'

    data = [context]

    return JsonResponse(data, safe=False)


def mapseoulpriceajax(request, id):
    print(id)
    print('<----------------------Ajax 통신')
    category = []
    location = []
    place = []
    year4mean = []
    martsi = []

    with open('./static/map_seoul_mean_price.csv', mode='r', encoding='utf-8') as seoul_lists:
        reader = csv.reader(seoul_lists)

        for list_num in reader:
            if list_num[0] == str(id):
                category.append(list_num[0])
                location.append(list_num[1])
                place.append(list_num[2])
                year4mean.append(int(round(float(list_num[3]),0)))
                martsi.append(list_num[4])

    pricelist = [[location[i], place[i], year4mean[i], martsi[i]] for i in range(len(martsi))]

    pricelistdesc = sorted(pricelist, key=lambda x:x[2])

    pricelistasc = sorted(pricelist, key=lambda x:-x[2])

    price_ex_location = [pricelistdesc[i][0] for i in range(3)]
    price_ex_place = [pricelistdesc[i][1] for i in range(3)]
    price_ex_price = [format(pricelistdesc[i][2],',') for i in range(3)]
    price_ex_martsi = [pricelistdesc[i][3] for i in range(3)]

    price_ch_location = [pricelistasc[i][0] for i in range(3)]
    price_ch_place = [pricelistasc[i][1] for i in range(3)]
    price_ch_price = [format(pricelistasc[i][2],',') for i in range(3)]
    price_ch_martsi = [pricelistasc[i][3] for i in range(3)]
    rank_num = [i for i in range(1,4)]

    print(rank_num, price_ex_location, price_ex_place, price_ex_price, price_ex_martsi)
    print('*'*100)
    print(rank_num, price_ch_location,price_ch_place, price_ch_price,  price_ch_martsi)


    context = {
        'priceExLocation':price_ex_location,
        'priceExPlace': price_ex_place,
        'priceExPrice': price_ex_price,
        'priceExMartsi': price_ex_martsi,

        'priceChLocation': price_ch_location,
        'priceChPlace': price_ch_place,
        'priceChPrice': price_ch_price,
        'priceChMartsi': price_ch_martsi,

        'length': len(price_ch_martsi),

        'rankNum': rank_num,

        'name': category[0]


    }
    data = [context]

    return JsonResponse(data, safe=False)

def vegetableSelectProducer(request, id):
    print(id)
    print('----------- ajax json vegetableSelectProducer')
    price_mart = []
    price_sijang = []
    category = []
    days = []

    trData = []
    martDic = {}
    sijangDic ={}

    fm_number = []
    fm_name = []
    fs_number = []
    fs_name = []



    # 변수 중요도 그래프 시장&마트
    with open('./static/feature_importance_시장_최종.csv', mode='r', encoding='utf-8-sig') as feature_lists_s:
        reader = csv.reader(feature_lists_s)
        for list_num in reader:
            if list_num[2] == str(id):
                fs_number.append(float(list_num[1]))
                fs_name.append(list_num[0])
                fm_number.append(float(list_num[3]))



    todayTest = datetime.today().strftime('%Y-%m-%d')
    yesterdayTest = (datetime.today()-timedelta(1)).strftime('%Y-%m-%d')
    twodaysagoTest = (datetime.today() - timedelta(2)).strftime('%Y-%m-%d')
    print(twodaysagoTest)

    with open('./static/sijang_pred_final.csv', mode='r', encoding='utf-8-sig') as vegetable_lists_p:
        reader = csv.reader(vegetable_lists_p)

        for list_num in reader:
            if list_num[2] == str(id):
                price_mart.append(int(list_num[3]))
                price_sijang.append(int(list_num[0]))
                category.append(list_num[2])
                days.append(list_num[1])

                martDic['kind'] = '마트'
                sijangDic['kind'] = '시장'

                weeklyMart1 = int(np.mean(price_mart))
                weeklyMart2 = int(np.trunc(weeklyMart1))
                martDic['weekly'] = weeklyMart2

                weeklySijang1 = int(np.mean(price_sijang))
                weeklySijang2 = int(np.trunc(weeklySijang1))
                sijangDic['weekly'] = weeklySijang2

                if list_num[1] == todayTest:
                    print("if1")
                    martDic['today'] = (int(list_num[3]))
                    sijangDic['today'] = (int(list_num[0]))

                if list_num[1] == yesterdayTest:
                    martDic['yesterday'] = (int(list_num[3]))
                    sijangDic['yesterday'] = (int(list_num[0]))
                    print("if2")

                if list_num[1] == twodaysagoTest:
                    martDic['twodaysago'] = (int(list_num[3]))
                    print("martDic['twodaysago']",martDic['twodaysago'])
                    sijangDic['twodaysago'] = (int(list_num[0]))
                    print("if3")

        martDic['gap'] = martDic['twodaysago'] - martDic['yesterday']
        sijangDic['gap'] = sijangDic['twodaysago'] - sijangDic['yesterday']

        martDic['weekly'] = format(martDic['weekly'],',')
        sijangDic['weekly'] = format(sijangDic['weekly'],',')
        martDic['today'] = format(martDic['today'],',')
        sijangDic['today'] = format(sijangDic['today'],',')
        martDic['yesterday'] = format(martDic['yesterday'],',')
        sijangDic['yesterday'] = format(sijangDic['yesterday'],',')
        martDic['twodaysago'] = format(martDic['twodaysago'],',')
        sijangDic['twodaysago'] = format(sijangDic['twodaysago'],',')
        martDic['gap'] = format(martDic['gap'],',')
        sijangDic['gap'] = format(sijangDic['gap'],',')

        trData.append(martDic)
        trData.append(sijangDic)

        print(">>>>>>>", trData)
        # print(">>>>>>>>", type(datetime.today().strftime('%Y-%m-%d')), datetime.today().strftime('%Y-%m-%d'))
        # print(">>>>>>>> list_num[1]  type: ", type(list_num[1]), list_num[1])


    context = {
        'price_mart': price_mart,
        'price_sijang' : price_sijang,
        'days': days,
        'category': category[0],

        'trData' : trData,
        'fm_number' : fm_number,
        'fm_name' : fm_name,
        'fs_number' : fs_number,
        'fs_name' : fs_name

    }
    data = []
    data.append(context)
    return JsonResponse(data, safe=False)

def get_map_kakao(request, id):
    print(id)
    ms = []
    location = []
    add = []
    tel = []
    place = []


    with open('./static/map_kakao.csv', mode='r', encoding='utf-8') as seoul_lists:
        reader = csv.reader(seoul_lists)

        for list_num in reader:
            if list_num[3] == id:
                place.append(list_num[0])
                ms.append(list_num[1])
                location.append(list_num[2])
                add.append(list_num[4])
                tel.append(list_num[5])

    context = {
        'place' : place,
        'ms': ms,
        'location': location,
        'add': add,
        'tel': tel,
        'length' : len(ms)
    }
    data = [context]
    return JsonResponse(data, safe=False)

def additionalfactors2(request):
    print('-------- ajax json additionalfactors2')

    days = []
    cabbage_price = []

    stock_seedkind = []
    stock_fertilizer = []
    stock_pesticide = []
    stock_machine = []
    stock_smartfarm = []

    with open('./static/stock_cabbagePrice.csv', mode='r', encoding='utf-8-sig') as stock_cabbagePrice:
        reader = csv.reader(stock_cabbagePrice)

        for list_num in reader:
            days.append(list_num[0])

            stock_seedkind.append(int(list_num[1]))
            stock_fertilizer.append(int(list_num[2]))
            stock_pesticide.append(int(list_num[3]))
            stock_machine.append(int(list_num[4]))
            stock_smartfarm.append(int(list_num[5]))

            cabbage_price.append(int(list_num[6]))

    context = {

        'days': days,
        'cabbage_price': cabbage_price,
        'stock_seedkind': stock_seedkind,
        'stock_fertilizer': stock_fertilizer,
        'stock_pesticide': stock_pesticide,
        'stock_machine': stock_machine,
        'stock_smartfarm': stock_smartfarm

    }
    data = [context]

    return JsonResponse(data, safe=False)

def search(request):
    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> search")

    item = request.POST['item']
    qty = request.POST['qty']

    # print(item , qty,' >>>item type:',type(item), '>>>qty type:',type(qty) )
    qty = int(qty)
    # print(item, qty, ' >>>item type:', type(item), '>>>qty type:', type(qty))
    predData = 0

    # 양파
    if (item=='1'):
        if(qty>101):
            predData = predData*0
        elif(qty>90):
            predData += 2848
        elif(qty>80):
            predData += 2833
        elif(qty>70):
            predData += 3067
        elif(qty>60):
            predData += 3049
        elif(qty>50):
            predData += 3495
        else :
            predData = predData*0

    # 배추
    if (item=='2'):
        if(qty>101):
            predData = predData*0
        elif(qty>90):
            predData += 2862
        elif(qty>80):
            predData += 4325
        elif(qty>70):
            predData += 3886
        elif(qty>60):
            predData += 3456
        elif(qty>50):
            predData += 3367
        else :
            predData = predData*0

    # 무
    if (item=='3'):
        if(qty>101):
            predData = predData*0
        elif(qty>90):
            predData += 1712
        elif(qty>80):
            predData += 1922
        elif(qty>70):
            predData += 1985
        elif(qty>60):
            predData += 1897
        elif(qty>50):
            predData += 1986
        else :
            predData = predData*0

    # 오이
    if (item=='4'):
        if(qty>101):
            predData = predData*0
        elif(qty>90):
            predData += 723
        elif(qty>80):
            predData += 755
        elif(qty>70):
            predData += 761
        elif(qty>60):
            predData += 701
        elif(qty>50):
            predData += 527
        else :
            predData = predData*0

    # 상추
    if (item=='5'):
        if(qty>101):
            predData = predData*0
        elif(qty>90):
            predData += 953
        elif(qty>80):
            predData += 1240
        elif(qty>70):
            predData += 1173
        elif(qty>60):
            predData += 1066
        elif(qty>50):
            predData += 887
        else :
            predData = predData*0

    data = [{'pred' : format(predData, ',') }]

    return JsonResponse(data , safe=False)

def predict(request):

    dummy1 = float(request.POST['dummy1'])
    dummy2 = float(request.POST['dummy2'])
    dummy3 = float(request.POST['dummy3'])
    dummy4 = float(request.POST['dummy4'])
    dummy5 = float(request.POST['dummy5'])
    dummy6 = float(request.POST['dummy6'])
    dummy7 = float(request.POST['dummy7'])
    dummy8 = float(request.POST['dummy8'])

    dummydf = pd.DataFrame({
        '경락가평균가격' : [dummy1],
        '반입량' :[dummy2],
        '유가 전국평균가격' : [dummy3],
        '유무':[dummy4],
        '최저기온(°C)':[dummy5],
        '최고기온(°C)':[dummy6],
        '일강수량(mm)':[dummy7],
        '도매가격': [dummy8]
    })

    dataset = pd.read_excel('./static/무_시장.xlsx', encoding='utf-8-sig')

    y_target = dataset['가격']
    X_data = dataset.drop(['가격'], axis=1, inplace=False)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_target, test_size=0.15, random_state=140)

    def get_model_cv_prediction(model, X_data, y_target):
        model.fit(X_data, y_target)

    xgb_reg = XGBRegressor(colsample_bytree=0.7,
                           gamma=0.1,
                           learning_rate=0.03,
                           max_depth=3,
                           min_child_weight=5,
                           n_estimators=200,
                           objective='reg:squarederror',
                           subsample=0.7)

    models = [xgb_reg]

    for model in models:
        get_model_cv_prediction(model, X_train, y_train)

    estimators = ([('xgb_reg', xgb_reg)])
    reg = StackingRegressor(estimators=estimators,
                            final_estimator=RandomForestRegressor(n_estimators=10, random_state=42))
    reg.fit(X_train, y_train).score(X_test, y_test)

    ypred1  = xgb_reg.predict(dummydf)

    ypred2 = int(round((-275.1522 + (dummy1 * 0.2110) + (dummy2 * 0.0005) + (dummy3 * 0.4948) + (dummy4 * 16.9529) + (
                dummy5 * 6.0531) + (dummy6 * -2.9252) + (dummy7 * 0.6929) + (dummy8 * 1.2675)), 0))

    data = [{
        'oneresult' : format(int(round(ypred1[0],0)),','),
        'tworesult' : format(ypred2,',')
    }]
    return JsonResponse(data, safe=False)

