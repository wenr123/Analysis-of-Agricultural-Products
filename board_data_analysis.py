import pandas as pd
import jieba
from wordcloud import WordCloud
import imageio
import missingno as msno
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import re
import webbrowser
import folium
from folium.plugins import HeatMap
import os
from tabao_mysql import Mysql_practice

mysql_mgr = Mysql_practice(4)

class T_dataanalysis:

    font = matplotlib.font_manager.FontProperties(fname='./taobao/data/DroidSansFallbackFull.ttf')

    def data_clean(self,datatmsp):
        msno.bar(datatmsp.sample(len(datatmsp)), figsize=(10, 4),color='purple')
        # 删除缺失值过半的列
        half_count = len(datatmsp) / 2
        datatmsp = datatmsp.dropna(thresh=half_count, axis=1)
        datatmsp = datatmsp.drop_duplicates()  # 删除重复行
        self.data = datatmsp[['title', 'province', 'region', 'discount_price', 'sale']]
        #print(self.data.head(10))

        self.title = self.data.title.values.tolist()

    def word_cloud(self,title = None):
        if title == None:
            title=self.title
        title_s = []
        for line in title:
            title_cut = jieba.lcut(line)
            title_s.append(title_cut)
        stopwords = pd.read_excel('./taobao/data/stopwords.xlsx')
        stopwords = stopwords.stopword.values.tolist()
        title_clean = []
        for line in title_s:
            line_clean = []
            for word in line:
                if word not in stopwords:
                    line_clean.append(word)
            title_clean.append(line_clean)
        # 去重，统计每个词语的个数
        #print(title_clean)
        self.title_clean_dist = []
        for line in title_clean:
            line_dist = []
            for words in line:
                #print(words)
                c = re.sub(" ", "", words)
                words = re.sub("[\d,500kg,'装']", "", c)
                if words =='':
                    continue
                else:
                    if words not in line_dist:
                        line_dist.append(words)
            self.title_clean_dist.append(line_dist)
        # 将所有次转换成list
        allwords_clean_dist = []
        for line in self.title_clean_dist:
            for word in line:
                allwords_clean_dist.append(word)
        # 将所有词语转换为数据表
        dr_allwords_clean_dist = pd.DataFrame({'allwords': allwords_clean_dist})
        self.word_count = dr_allwords_clean_dist.allwords.value_counts().reset_index()
        self.word_count.columns = ['word', 'count']
        print("title关键词统计")
        print(self.word_count.head(10))
        print("-------------------------------------------")
        plt.figure(figsize=(20, 10))
        pic = imageio.imread("./taobao/image/agricultural_products.jpg")
        a_g = WordCloud(font_path="./taobao/data/DroidSansFallbackFull.ttf", background_color='white',
                        mask=pic,max_font_size=60,margin=1)
        ag = a_g.fit_words({x[0]: x[1] for x in self.word_count.head(100).values})
        plt.imshow(ag, interpolation='bilinear')
        plt.show()

    def key_statistics(self,word_count=None,data=None):
        if word_count==None:
            word_count = self.word_count
        if data==None:
            data = self.data
        w_s_sum = []
        for w in word_count.word:
            i = 0
            s_list = []
            for t in self.title_clean_dist:
                if w in t:
                    try:
                        s_list.append(data.sale[i])
                    except:
                        s_list.append(0)
                i += 1
            w_s_sum.append(sum(s_list))
        df_w_s_sum = pd.DataFrame({'w_s_sum': w_s_sum})
        # print(word_count.head())
        # print(df_w_s_sum.head())
        df_word_sum = pd.concat([word_count, df_w_s_sum], axis=1, ignore_index=True)
        df_word_sum.columns = ['word', 'count', 'w_s_sum']
        print("出现次数排名前20的title关键词统计")
        print(df_word_sum.head(20))
        df_word_sum.sort_values('w_s_sum', inplace=True, ascending=True)
        df_w_s = df_word_sum.tail(15)
        #font = matplotlib.font_manager.FontProperties(fname='./taobao/data/DroidSansFallbackFull.ttf')
        index = np.arange(df_w_s.word.size)
        plt.figure(figsize=(6, 12))
        plt.barh(index,df_w_s.w_s_sum,color='purple',align='center',alpha=0.8)
        plt.yticks(index, list(df_w_s.word), fontproperties=self.font)
        for y, x in zip(index, df_w_s.w_s_sum):
            plt.text(x, y, "%.0f" % x, ha='left', va='center')
        plt.title(u"排名前15销量的关键词", fontsize=15, fontproperties=self.font)
        plt.show()

    def pathmark(self,data=None):
        if data==None:
            data = self.data
        data_p = data[data['discount_price'] < 200]

        plt.figure(figsize=(7, 5))
        plt.hist(data_p['discount_price'], bins=20, color='purple')
        plt.xlabel(u"价格", fontsize=12, fontproperties=self.font)
        plt.ylabel(u"商品数量", fontsize=12, fontproperties=self.font)
        plt.title(u"不同价格对应的商品数量分布", fontsize=15, fontproperties=self.font)
        # plt.show()

        # 不同价格区间的商品的平均销量分布：
        data['price'] = data.discount_price.astype('int')
        data['group'] = pd.qcut(data.price, 10)  # 将price列分为10组
        dr_group = data.group.value_counts().reset_index()
        # 以group列进行分类求sales的均值
        df_s_g = data[['sale', 'group']].groupby('group').mean().reset_index()

        # 柱型图
        index = np.arange(df_s_g.group.size)
        plt.figure(figsize=(8, 4))
        plt.bar(index, df_s_g.sale, color='purple')
        plt.xticks(index, df_s_g.group, fontsize=8, rotation=30)
        plt.xlabel('group')
        plt.ylabel('mean_sales')
        plt.title('不同价格区间的商品平均销量', fontproperties=self.font)

        # 商品价格对销量的影响分析
        fig, ax = plt.subplots()
        ax.scatter(data_p['discount_price'], data_p['sale'], color='purple')
        ax.set_xlabel('价格', fontproperties=self.font)
        ax.set_ylabel('销量', fontproperties=self.font)
        ax.set_title('商品价格对销量的影响', fontproperties=self.font)
        plt.show()
        # 商品价格对销售额的影响分析
        # data['GMV'] = data['price'] * data['sale']
        # sns.regplot(x="price", y='GMV', data=data, color='purple')
        # plt.show()

    def good_distribution(self,data=None):
        if data == None:
            data = self.data
        # 不同省份的商品数量分布
        plt.figure(figsize=(8, 4))
        data.province.value_counts().plot(kind='bar', color='purple')
        plt.xticks(rotation=45, fontsize=1, fontproperties=self.font)
        plt.xlabel('省份', fontproperties=self.font)
        plt.ylabel('数量', fontproperties=self.font)
        plt.title('不同省份的商品数量分布', fontproperties=self.font)
        # 不同省份的商品平均销量分布
        pro_sales = data.pivot_table(index='province', values='sale', aggfunc=np.mean)  # 分类求均值
        pro_sales.sort_values('sale', inplace=True, ascending=False)  # 排序
        pro_sales = pro_sales.reset_index()
        index = np.arange(pro_sales.sale.size)
        plt.figure(figsize=(8, 6))
        plt.bar(index, pro_sales.sale, color='purple')
        plt.xticks(index, pro_sales.province, fontsize=1, rotation=45, fontproperties=self.font)
        plt.xlabel('province', fontproperties=self.font)
        plt.ylabel('mean_sales', fontproperties=self.font)
        plt.title('不同省份的商品平均销量分布', fontproperties=self.font)
        plt.show()
        pro_sales.to_excel('./taobao/data/pro_sales.xlsx', index=False)

    # def heat_map(self,pro_sales=None):
    #     if pro_sales == None:
    #         pro_sales = self.pro_sales
    #     heat_data = [pro_sales.sale,pro_sales.province]
    #     m = folium.Map([33., 113.], tiles='stamentoner', zoom_start=5)
    #     HeatMap(heat_data).add_to(m)
    #     m.save(os.path.join(r'F:\Pythonpractice\learn\crawl_practice\paper_design\board_data_analysis\taobao\data', 'Heatmap.html'))

if __name__ == '__main__':
    datatmsp = pd.read_csv('./taobao/data/new_board1.csv', encoding='ANSI')
    plc = T_dataanalysis()
    plc.data_clean(datatmsp)
    plc.word_cloud()
    plc.key_statistics()
    plc.pathmark()
    plc.good_distribution()
