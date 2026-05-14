
#导入包

#数据处理
import pandas as pd
#数据分割
from sklearn.model_selection import train_test_split
#数据标准化加工
from sklearn.preprocessing import StandardScaler
#实例化模型
from sklearn.tree import DecisionTreeClassifier
#模型评分
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#可视化处理
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
#导出模型
import joblib


def trans_sex(x) -> int:
    if x == 'male':
        return 1
    else:
        return 0

def train_model(model_path : str = 'model', save : bool = False):

    #导入本地数据
    data = pd.read_csv('tested.csv')

    #对空值进行处理
    data['Age'] = data['Age'].fillna(value = data['Age'].mean())

    #数据选择
    x = data[['Pclass', 'Age', 'Sex']].copy()
    y = data['Survived']

    x['Sex'] = x['Sex'].apply(func = trans_sex)

    #数据集分割
    x_train, x_text, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)

    #数据标准化
    transformer = StandardScaler()
    x_train = transformer.fit_transform(x_train)
    x_text = transformer.transform(x_text)

    #模型训练
    estimator = DecisionTreeClassifier(criterion = 'gini', max_depth=4, min_samples_leaf=5, random_state=22)   #默认使用CART决策树
    estimator.fit(x_train, y_train)

    #模型预测
    y_predict = estimator.predict(x_text)

    #模型评分
    print('accuracy_score-->', accuracy_score(y_test, y_predict))

    #模型具体报告
    print('classification_report-->\n', classification_report(y_test, y_predict, target_names = ['Died', 'Survived']))


    #进行可视化处理
    plot_tree(estimator,
        max_depth = 10,
        filled = True,
        feature_names = ['Pclass', 'Age', 'Sex_0', 'Sex_1'],
        class_names = ['Died', 'Survived']
    )

    plt.show()

    if save:
        joblib.dump(estimator, model_path)


if __name__ == '__main__':
    train_model()
