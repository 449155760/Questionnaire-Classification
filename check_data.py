import pandas as pd
import numpy as np


def check_statistics_sheet(filepath='data.xlsx'):
    """检查统计数据sheet"""
    print(f"检查文件: {filepath}")

    # 查看所有sheet
    print("=" * 60)
    print("Excel文件中的所有sheet:")
    xl = pd.ExcelFile(filepath)
    sheet_names = xl.sheet_names
    print(f"Sheet名称: {sheet_names}")

    if '统计数据' in sheet_names:
        sheet_name = '统计数据'
    else:
        sheet_name = sheet_names[0]  # 使用第一个sheet

    print(f"正在读取sheet: '{sheet_name}'")
    print("=" * 60)

    # 读取数据
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    print(f"数据形状: {df.shape} (行×列)")
    print(f"列名 ({len(df.columns)}个):")
    for i, col in enumerate(df.columns):
        print(f"  {i:3d}. {col}")

    # 查看前5行数据
    print("\n前5行数据:")
    print(df.head())

    # 检查后5行数据
    print("\n后5行数据:")
    print(df.tail())

    # 检查目标变量列
    target_cols = [col for col in df.columns if '剖面' in str(col) or '分类' in str(col)]
    print(f"\n可能的目标变量列: {target_cols}")

    if target_cols:
        target_col = target_cols[0]
        print(f"\n使用目标变量列: '{target_col}'")
        print(f"目标变量数据类型: {df[target_col].dtype}")
        print(f"目标变量唯一值: {sorted(df[target_col].dropna().unique())}")
        print(f"目标变量值分布:\n{df[target_col].value_counts().sort_index()}")
        print(f"目标变量缺失值: {df[target_col].isnull().sum()}")
    else:
        print("警告: 未找到明确的目标变量列")
        # 显示所有列名的前几个字符
        print("所有列名:")
        for col in df.columns:
            print(f"  '{col}'")

    # 检查数据类型
    print("\n数据类型统计:")
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"  {dtype}: {count} 列")

    # 检查缺失值
    print("\n缺失值统计:")
    missing_info = df.isnull().sum()
    missing_cols = missing_info[missing_info > 0]

    if len(missing_cols) > 0:
        print(f"有 {len(missing_cols)} 列存在缺失值:")
        for col, missing in missing_cols.items():
            percentage = missing / len(df) * 100
            print(f"  {col}: {missing} 个缺失值 ({percentage:.1f}%)")
    else:
        print("没有缺失值")

    # 检查数值特征
    print("\n数值特征统计:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"数值列数量: {len(numeric_cols)}")

    # 显示重要数值列的基本统计
    important_numeric = []
    for col in numeric_cols:
        if any(keyword in str(col) for keyword in
               ['年龄', 'BMI', 'ADL', '时间', '评分', '总分', '平均分', '状态', '能力', '支持', '量表', '质量']):
            important_numeric.append(col)

    print("重要数值特征:")
    for col in important_numeric[:20]:  # 限制显示数量
        if df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            missing = df[col].isnull().sum()
            print(f"  {col}: 范围 [{min_val:.2f}, {max_val:.2f}], 均值={mean_val:.2f}, 缺失={missing}")

    # 检查分类特征
    print("\n分类特征统计:")
    categorical_cols = []
    for col in df.columns:
        if col not in numeric_cols or (col in numeric_cols and df[col].nunique() <= 10):
            if df[col].nunique() <= 20:  # 唯一值少于20个
                categorical_cols.append(col)

    print(f"可能的分类特征数量: {len(categorical_cols)}")
    important_categorical = []
    for col in categorical_cols:
        if any(keyword in str(col) for keyword in
               ['性别', '文化', '婚姻', '居住', '工作', '医保', '收入', '诊断', '手术', '康复']):
            important_categorical.append(col)

    print("重要分类特征:")
    for col in important_categorical[:15]:  # 限制显示数量
        unique_vals = sorted(df[col].dropna().unique())
        nunique = len(unique_vals)
        missing = df[col].isnull().sum()
        if nunique <= 10:
            print(f"  {col}: {nunique} 个唯一值 - {unique_vals}, 缺失={missing}")
        else:
            print(f"  {col}: {nunique} 个唯一值, 缺失={missing}")

    # 检查出院准备度相关特征
    print("\n出院准备度相关特征:")
    prep_cols = []
    for col in df.columns:
        if any(keyword in str(col) for keyword in
               ['出院准备度', '个人状态', '适应能力', '预期性支持', '总分', '平均分']):
            prep_cols.append(col)

    for col in prep_cols:
        if col in df.columns:
            if col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                missing = df[col].isnull().sum()
                print(f"  {col}: 范围 [{min_val:.2f}, {max_val:.2f}], 均值={mean_val:.2f}, 缺失={missing}")
            else:
                print(f"  {col}: {df[col].dtype}, 唯一值数={df[col].nunique()}")

    # 保存列名列表
    with open('statistics_columns.txt', 'w', encoding='utf-8') as f:
        f.write("统计数据sheet列名列表:\n")
        for col in df.columns:
            f.write(f"{col}\n")

    print(f"\n列名列表已保存到 statistics_columns.txt")

    return df, target_cols


if __name__ == "__main__":
    filename = 'data.xlsx'

    try:
        df, target_cols = check_statistics_sheet(filename)

        # 目标变量详细分析
        if target_cols:
            target_col = target_cols[0]

            print("\n" + "=" * 60)
            print("目标变量详细分析:")
            print("=" * 60)

            # 统计每个类别的样本数量
            class_counts = df[target_col].value_counts().sort_index()
            print(f"类别分布:")
            for class_val, count in class_counts.items():
                percentage = count / len(df) * 100
                print(f"  类别 {class_val}: {count} 个样本 ({percentage:.1f}%)")

            # 检查类别平衡
            if len(class_counts) > 0:
                min_count = class_counts.min()
                max_count = class_counts.max()
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                print(f"类别不平衡比例: {imbalance_ratio:.2f}")

                if imbalance_ratio > 2:
                    print("注意: 存在类别不平衡，建议在模型训练中使用 class_weight='balanced'")

            # 检查目标变量类型
            print(f"\n目标变量类型: {df[target_col].dtype}")
            print(f"目标变量示例值: {df[target_col].iloc[:5].tolist()}")

        # 建议用于建模的特征
        print("\n" + "=" * 60)
        print("建议用于建模的特征:")
        print("=" * 60)

        # 推荐的特征分类
        recommended_features = {
            "人口学特征": ["性别", "年龄", "BMI", "文化程度", "婚姻状况", "居住方式", "工作状态",
                           "医保类型", "家庭收入情况", "居住地"],
            "临床特征": ["主诊断", "手术方式", "慢病共存", "合并骨松", "康复介入",
                         "出院时ADL评分", "住院时间", "BI"],
            "量表评分": ["社会支持评定量表", "出院指导质量"],
            "出院准备度核心特征": ["个人状态(1-3)", "适应能力(4-8)", "预期性支持(9-12)"],
            "其他特征": ["出院准备度总分", "出院准备度平均分"]
        }

        # 检查哪些特征存在
        existing_features = {}
        for category, features in recommended_features.items():
            existing = [f for f in features if f in df.columns]
            if existing:
                existing_features[category] = existing

        print("建议使用的特征（基于您的文档）:")
        for category, features in existing_features.items():
            print(f"\n{category} ({len(features)}个):")
            for feature in features:
                print(f"  - {feature}")

        print(f"\n总建议特征数: {sum(len(f) for f in existing_features.values())}")

    except Exception as e:
        print(f"检查数据时出错: {e}")
        import traceback

        traceback.print_exc()