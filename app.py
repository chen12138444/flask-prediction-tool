#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
import lime
import lime.lime_tabular

# 创建Flask应用
app = Flask(__name__)

# 读取开发数据集并训练模型
development_data = pd.read_csv("E:/main/web/com5.csv")

# 选择自变量和因变量
X = development_data[['AGE', 'AMH', 'FSH', 'TREATMENT', 'UTERINE_INFERTILITY']]
y = development_data['LIVE_BIRTH_OR_NOT']

# 构建并训练逻辑回归模型
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# LIME解释器设置
feature_names = ['Age', 'AMH', 'FSH', 'Treatment', 'Uterine infertility']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 获取表单输入
    age = float(request.form['age'])
    amh = float(request.form['amh'])
    fsh = float(request.form['fsh'])
    treatment = int(request.form['treatment'])  # 2 for IVF, 1 for ICSI
    uterine_infertility = int(request.form['uterine_infertility'])  # 1 for Yes, 0 for No
    
    # 准备输入数据进行预测
    input_data = pd.DataFrame([[age, amh, fsh, treatment, uterine_infertility]], 
                              columns=['AGE', 'AMH', 'FSH', 'TREATMENT', 'UTERINE_INFERTILITY'])
    
    # 预测
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # 使用LIME解释器生成解释
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values, 
        feature_names=feature_names, 
        class_names=['Not Live Birth', 'Live Birth'], 
        discretize_continuous=True
    )
    
    exp = explainer.explain_instance(input_data.values[0], model.predict_proba, num_features=5)
    lime_html = exp.as_html()

    return render_template('result.html', 
                           prediction=prediction[0], 
                           prediction_proba=prediction_proba[0], 
                           lime_html=lime_html)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', use_reloader=False, port=5000)


# In[ ]:


get_ipython().system('python app.ipynb')


# In[ ]:




