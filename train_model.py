import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置环境变量，避免中文路径问题
os.environ['LOKY_MAX_CPU_COUNT'] = '2'
os.environ['JOBLIB_TEMP_FOLDER'] = './temp'


def load_data(filepath='data.xlsx'):
    """
    加载统计数据sheet
    """
    print("正在加载数据...")

    # 读取统计数据sheet
    try:
        df = pd.read_excel(filepath, sheet_name='统计数据')
    except:
        # 如果sheet名不对，读取第一个sheet
        df = pd.read_excel(filepath, sheet_name=0)

    print(f"数据形状: {df.shape}")
    print(f"样本数: {len(df)}")

    # 目标变量
    target_col = '出院准备度剖面分类'
    if target_col not in df.columns:
        raise ValueError(f"未找到目标变量列: {target_col}")

    y = df[target_col]
    X = df.drop(columns=[target_col, '编号'])  # 移除编号列

    print(f"\n目标变量分布:")
    print(y.value_counts().sort_index())
    print(f"类别比例: 1:{sum(y == 1)} 2:{sum(y == 2)} 3:{sum(y == 3)}")

    return X, y


def select_features(X, y):
    """
    选择用于建模的特征
    """
    print("\n选择建模特征...")

    # 所有重要特征
    important_features = [
        # 人口学特征
        '性别', '年龄', 'BMI', '文化程度', '婚姻状况', '居住方式',
        '工作状态', '医保类型', '家庭收入情况', '居住地',
        # 临床特征
        '主诊断', '手术方式', '慢病共存', '合并骨松', '康复介入',
        'BI', '住院时间',
        # 量表评分
        '社会支持评定量表', '出院指导质量',
        # 出院准备度核心维度
        '个人状态(1-3)', '适应能力(4-8)', '预期性支持(9-12)',
        # 出院准备度汇总
        '出院准备度总分', '出院准备度平均分'
    ]

    # 检查哪些特征存在
    existing_features = [f for f in important_features if f in X.columns]
    print(f"使用 {len(existing_features)} 个特征")

    # 使用所有存在的特征
    X_selected = X[existing_features].copy()

    return X_selected, existing_features


def analyze_features(X, y):
    """
    分析特征类型和分布
    """
    print("\n分析特征类型...")

    # 识别特征类型
    categorical_features = []
    numerical_features = []

    for col in X.columns:
        # 如果是数值类型且唯一值数量少，可能是分类变量
        if pd.api.types.is_numeric_dtype(X[col]):
            unique_count = X[col].nunique()
            if unique_count <= 10 and col not in ['年龄', 'BMI', '住院时间', '社会支持评定量表',
                                                  '出院指导质量', '个人状态(1-3)', '适应能力(4-8)',
                                                  '预期性支持(9-12)', '出院准备度总分', '出院准备度平均分', 'BI']:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        else:
            categorical_features.append(col)

    print(f"分类特征 ({len(categorical_features)}个):")
    for feat in categorical_features:
        print(f"  - {feat}: {X[feat].nunique()} 个唯一值")

    print(f"\n数值特征 ({len(numerical_features)}个):")
    for feat in numerical_features:
        print(f"  - {feat}: 范围 [{X[feat].min():.1f}, {X[feat].max():.1f}]")

    return categorical_features, numerical_features


def create_preprocessing_pipeline(categorical_features, numerical_features):
    """
    创建预处理管道
    """
    print("\n创建预处理管道...")

    # 数值特征处理：标准化
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # 分类特征处理：独热编码
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'))
    ])

    # 合并预处理步骤
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return preprocessor


def train_and_evaluate_model(X, y, categorical_features, numerical_features):
    """
    训练和评估模型
    """
    print("\n" + "=" * 60)
    print("训练和评估模型")
    print("=" * 60)

    # 创建预处理管道
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"训练集: {X_train.shape[0]} 个样本")
    print(f"测试集: {X_test.shape[0]} 个样本")

    # 创建最终模型管道 - 使用简化参数避免并行计算
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            n_jobs=1  # 单线程，避免并行问题
        ))
    ])

    # 训练模型
    print("\n训练模型中...")
    model.fit(X_train, y_train)

    # 评估模型
    print("\n模型评估:")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"测试集准确率: {accuracy:.4f}")
    print(f"测试集F1分数: {f1:.4f}")

    # 分类报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['类别1', '类别2', '类别3']))

    # 混淆矩阵
    print("混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return model, preprocessor, X_test, y_test, accuracy, f1


def train_final_model(X, y, categorical_features, numerical_features):
    """
    在完整数据集上训练最终模型
    """
    print("\n" + "=" * 60)
    print("在完整数据集上训练最终模型")
    print("=" * 60)

    # 创建预处理管道
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)

    # 创建最终模型
    final_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=42,
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            n_jobs=1
        ))
    ])

    # 在完整数据集上训练
    print("训练最终模型...")
    final_model.fit(X, y)

    # 交叉验证评估
    print("进行5折交叉验证...")
    cv_scores = cross_val_score(final_model, X, y, cv=5, scoring='f1_weighted', n_jobs=1)
    print(f"5折交叉验证F1分数: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    return final_model, preprocessor, cv_scores.mean()


def save_model(model, preprocessor, feature_names, categorical_features, numerical_features, cv_score):
    """
    保存模型和相关信息
    """
    import os

    # 创建model目录
    os.makedirs('model', exist_ok=True)

    # 保存模型
    model_path = 'model/model.pkl'
    joblib.dump(model, model_path)
    print(f"模型已保存到 {model_path}")

    # 保存预处理管道
    preprocessor_path = 'model/preprocessor.pkl'
    joblib.dump(preprocessor, preprocessor_path)
    print(f"预处理管道已保存到 {preprocessor_path}")

    # 保存特征信息
    model_info = {
        'feature_names': feature_names,
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'target_classes': [1, 2, 3],
        'model_type': 'RandomForest',
        'version': '1.0',
        'description': '出院准备度剖面分类预测模型',
        'data_source': '统计数据sheet',
        'cv_score': cv_score,
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    model_info_path = 'model/model_info.pkl'
    joblib.dump(model_info, model_info_path)
    print(f"模型信息已保存到 {model_info_path}")

    # 保存特征列表
    feature_list_path = 'model/feature_list.txt'
    with open(feature_list_path, 'w', encoding='utf-8') as f:
        f.write("用于建模的特征列表\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型版本: 1.0\n")
        f.write(f"训练日期: {model_info['training_date']}\n")
        f.write(f"交叉验证F1分数: {cv_score:.4f}\n")
        f.write(f"总特征数: {len(feature_names)}\n")
        f.write(f"分类特征: {len(categorical_features)}\n")
        f.write(f"数值特征: {len(numerical_features)}\n\n")

        f.write("所有特征:\n")
        for i, feature in enumerate(feature_names, 1):
            f.write(f"  {i:2d}. {feature}\n")

        f.write("\n分类特征:\n")
        for i, feature in enumerate(categorical_features, 1):
            f.write(f"  {i:2d}. {feature}\n")

        f.write("\n数值特征:\n")
        for i, feature in enumerate(numerical_features, 1):
            f.write(f"  {i:2d}. {feature}\n")

    print(f"特征列表已保存到 {feature_list_path}")

    return model_path, preprocessor_path, model_info_path


def create_test_template(feature_names):
    """
    创建测试数据模板
    """
    template_path = 'test_template.xlsx'

    # 创建示例数据
    template_data = {}
    for feature in feature_names:
        if feature == '性别':
            template_data[feature] = [1, 2]  # 1:男, 2:女
        elif feature == '年龄':
            template_data[feature] = [70, 75]
        elif feature == 'BMI':
            template_data[feature] = [1, 1]  # 1:正常
        elif feature in ['文化程度', '婚姻状况', '居住方式', '工作状态', '医保类型',
                         '家庭收入情况', '居住地', '主诊断', '手术方式', '慢病共存',
                         '合并骨松', '康复介入']:
            template_data[feature] = [1, 1]  # 默认值
        elif feature == 'BI':
            template_data[feature] = [60, 65]
        elif feature == '住院时间':
            template_data[feature] = [10, 12]
        elif feature == '社会支持评定量表':
            template_data[feature] = [30, 35]
        elif feature == '出院指导质量':
            template_data[feature] = [120, 130]
        elif feature == '个人状态(1-3)':
            template_data[feature] = [15, 16]
        elif feature == '适应能力(4-8)':
            template_data[feature] = [25, 26]
        elif feature == '预期性支持(9-12)':
            template_data[feature] = [30, 32]
        elif feature == '出院准备度总分':
            template_data[feature] = [70, 74]
        elif feature == '出院准备度平均分':
            template_data[feature] = [5.8, 6.2]

    # 创建DataFrame
    df_template = pd.DataFrame(template_data)

    # 添加编号
    df_template.insert(0, '编号', [1, 2])

    # 保存模板
    df_template.to_excel(template_path, index=False)
    print(f"\n测试数据模板已创建: {template_path}")
    print("请使用此模板准备测试数据")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("出院准备度剖面分类模型训练系统 (修复版)")
    print("=" * 60)

    try:
        # 1. 加载数据
        X, y = load_data('data.xlsx')

        # 2. 选择特征
        X_selected, feature_names = select_features(X, y)

        # 3. 分析特征类型
        categorical_features, numerical_features = analyze_features(X_selected, y)

        # 4. 训练和评估模型
        model, preprocessor, X_test, y_test, accuracy, f1 = train_and_evaluate_model(
            X_selected, y, categorical_features, numerical_features
        )

        # 5. 在完整数据集上训练最终模型
        final_model, final_preprocessor, cv_score = train_final_model(
            X_selected, y, categorical_features, numerical_features
        )

        # 6. 保存模型
        model_path, preprocessor_path, model_info_path = save_model(
            final_model, final_preprocessor, feature_names,
            categorical_features, numerical_features, cv_score
        )

        print("\n" + "=" * 60)
        print("模型训练完成！")
        print("=" * 60)

        # 7. 创建测试模板
        create_test_template(feature_names)

        # 使用说明
        print("\n使用说明:")
        print("1. 运行 predict_final.py 启动预测工具")
        print("2. 使用 test_template.xlsx 作为模板准备测试数据")
        print("3. 确保测试数据包含所有24个特征")
        print(f"4. 模型性能:")
        print(f"   - 交叉验证F1分数: {cv_score:.4f}")
        print(f"   - 测试集准确率: {accuracy:.4f}")
        print(f"   - 测试集F1分数: {f1:.4f}")

        print("\n5. 模型文件已保存到 model/ 目录:")
        print(f"   - 模型文件: {model_path}")
        print(f"   - 预处理管道: {preprocessor_path}")
        print(f"   - 模型信息: {model_info_path}")

    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置工作目录为当前脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()