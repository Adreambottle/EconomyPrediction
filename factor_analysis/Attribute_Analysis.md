


* Logic_flag: 同向为1,反向为-1。由于该字段只有两个字符串值，同向和反向，因此直接用==逻辑判断就可以

```python
df['Logic_flag'] = (df['flag'] == '同向')
df['Logic_flag'].replace({True: 1, False: -1}, inplace=True)
```


* Var_flag
创建 Var_flag 原则上该字段不需要考虑Y的变化，但是为了保证同向的可解释性，得判断X和Y的变化一致性，
默认Y上升为1，下降为-1，持平和未知为0

首先，将X和Y的状态转化为统一的数字，因为下降和上升一共有6种描述，再加上两种原始描述（上升、下降）
一共有8种描述变化的字符串


```python
df['Var_flag'] = df['x_status_db'].replace({'上升': 1, '下降': -1, '持平': 0, '未知': np.nan})
```

* Final_flag

创建Final_flag，X 与 Y最终方向一致，Final_flag = Var_flag * Logic_flag

```python
df['Final_flag'] = df['Var_flag'] * df['Logic_flag']
```

'''
读取回溯期内指标历史数据，利用指标回溯期变化率给本期变化率打分
'''


* 归因指标
GDP

* 作为解释变量

x_index
```
['规模以上工业增加值', '先进制造业增加值', '高技术制造业增加值', '先进轻纺制造业增加值', '先进装备制造业增加值',
 '家用电力器具制造业增加值', '电子及通信设备制造业增加值', '高端电子信息制造业增加值', '金属制品业增加值',
 '新材料制造业增加值', '石油化工产业增加值', '纺织服装增加值', '建筑材料增加值', '产成品存货', ... ]
```

人工的先验知识


``` 
 {'id': nan,
  'kb_tag': 'GDP',
  'index_y_id': 'IDX001',
  'index_y': '地区生产总值',
  'index_x': '规模以上工业增加值',
  'index_x_id': 'IDX002',
  'tag1': '第二产业',
  'tag2': '工业',
  'flag': '同向',
  'affect_way': '组成',
  'period': 0,
  'x_info': '是反映中山地区工业发展情况的重要指标',
  'x_status': '下降',
  'y_status': '降幅扩大',
  'precedent_logic': nan,
  'status_logic': '直接表明工业企业整体效益不佳',
  'affect_logic': '对GDP增速恶化的影响较为显著',
  'full_info': nan,
  'relation_area': '中山',
  'classic_flag': 0.8,
  'tag': nan,
  'batch_no': nan},
```

4: {'id': nan,
  'kb_tag': 'GDP',
  'index_y_id': 'IDX001',
  'index_y': '地区生产总值',
  'index_x': '规模以上工业增加值',
  'index_x_id': 'IDX002',
  'tag1': '第二产业',
  'tag2': '工业',
  'tag3': nan,
  'flag': '同向',
  'affect_way': '组成',
  'period': 0,
  'x_info': '是反映中山地区工业发展情况的重要指标',
  'x_status': '下降',
  'y_status': '增速转负',
  'precedent_logic': nan,
  'status_logic': '直接表明工业企业整体效益不佳',
  'affect_logic': '对GDP增速造成的负面影响较为显著',
  'full_info': nan,
  'info_catagory': nan,
  'info_source': nan,
  'pub_date': nan,
  'relation_area': '中山',
  'classic_flag': 0.8,
  'tag': nan,
  'batch_no': nan},