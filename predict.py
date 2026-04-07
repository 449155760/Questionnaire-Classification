import sklearn.ensemble
import sklearn.pipeline
import sklearn.compose
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.tree
import openpyxl  # [重要修复]：显式引用 openpyxl，防止 EXE 运行时 pandas 无法读写 Excel
# 以上是新增的显式引用，专门服务于 PyInstaller 打包

import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def get_resource_path(relative_path):
    """获取资源绝对路径，兼容 PyInstaller 打包后的路径"""
    if getattr(sys, 'frozen', False):
        # 如果是打包后的 EXE 运行，资源在 sys._MEIPASS 临时目录下
        base_path = sys._MEIPASS
    else:
        # 如果是普通 python 脚本运行，资源在当前文件所在目录下
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def get_executable_path():
    """获取 EXE 文件所在的物理路径，用于存放生成的 Excel 结果"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def main():
    """主函数"""
    print("=" * 60)
    print("出院准备度剖面分类预测工具 v1.0")
    print("=" * 60)

    # 1. 定位模型文件夹路径 (使用兼容函数)
    model_dir = get_resource_path('model')

    # 2. 设置工作目录为 EXE 所在目录 (方便用户直接输入文件名)
    exe_dir = get_executable_path()
    os.chdir(exe_dir)

    # 加载模型
    try:
        print(f"正在加载模型资源...")

        # 检查必要文件
        required_files = ['model.pkl', 'preprocessor.pkl', 'model_info.pkl']
        for file in required_files:
            file_path = os.path.join(model_dir, file)
            if not os.path.exists(file_path):
                print(f"❌ 错误: 内部资源缺失 {file}")
                input("按回车键退出...")
                return

        # 加载模型文件
        model = joblib.load(os.path.join(model_dir, 'model.pkl'))
        model_info = joblib.load(os.path.join(model_dir, 'model_info.pkl'))

        # 注意: 你的代码里检查了 preprocessor.pkl，如果你的模型是 pipeline，则预处理已经包含在 model 中；
        # 如果训练时是分离保存的，且预测需要预处理，记得在这里也 joblib.load 并在 predict_file 中使用。
        # 既然你说之前的 Python 环境跑通了，这里逻辑保持不变。

        print("✅ 模型加载成功")
        print(f"模型类型: {model_info.get('model_type', '未知')}")
        print(f"特征数量: {len(model_info.get('feature_names', []))}")

    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        input("\n按回车键退出...")
        return

    while True:
        print("\n" + "=" * 40)
        print("请选择操作:")
        print("1. 预测单个文件")
        print("2. 查看模型信息")
        print("3. 退出")

        try:
            choice = input("\n请输入选择 (1/2/3): ").strip()

            if choice == '1':
                # 兼容拖拽文件带来的双引号或多余空格
                filepath = input("请输入测试文件路径 (或直接将文件拖入此处): ").strip(' "\'')

                if not os.path.exists(filepath):
                    print("❌ 文件不存在，请检查路径是否正确。")
                    continue

                if not filepath.lower().endswith(('.xlsx', '.xls')):
                    print("❌ 请选择Excel文件 (.xlsx 或 .xls)")
                    continue

                predict_file(filepath, model, model_info)

            elif choice == '2':
                show_model_info(model_info)

            elif choice == '3':
                print("谢谢使用，再见！")
                import time
                time.sleep(1)  # 增加小停顿，让体验更好
                break
            else:
                print("❌ 无效选择，请重新输入。")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")


def predict_file(filepath, model, model_info):
    """预测单个文件"""
    try:
        print(f"\n正在处理: {os.path.basename(filepath)}")

        # 1. 加载测试数据
        test_df = pd.read_excel(filepath)

        # 2. 验证特征
        required_features = model_info.get('feature_names', [])
        missing_features = [f for f in required_features if f not in test_df.columns]

        if missing_features:
            print(f"❌ 错误：Excel中缺少以下必要列:\n{missing_features}")
            return

        # 3. 提取特征并转换类型
        X_test = test_df[required_features].copy()
        for col in X_test.columns:
            X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

        # 4. 执行预测
        print("计算中...")
        predictions = model.predict(X_test)

        # 5. 组装结果
        result_df = test_df.copy()
        result_df['预测类别'] = predictions

        class_descriptions = {
            1: "低准备度",
            2: "中准备度",
            3: "高准备度"
        }
        result_df['预测类别描述'] = result_df['预测类别'].map(class_descriptions)

        # 6. 打印统计
        print("\n预测结果汇总:")
        unique, counts = np.unique(predictions, return_counts=True)
        for pred_type, count in zip(unique, counts):
            desc = class_descriptions.get(pred_type, f"类别{pred_type}")
            print(f"  - {desc}: {count}人")

        # 7. 保存结果到原文件所在目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.dirname(os.path.abspath(filepath))
        output_name = f"预测结果_{timestamp}.xlsx"
        output_path = os.path.join(output_dir, output_name)

        result_df.to_excel(output_path, index=False)
        print(f"\n✅ 成功！结果已保存至:\n{output_path}")

    except Exception as e:
        print(f"❌ 预测过程出错: {str(e)}")


def show_model_info(model_info):
    """显示模型信息"""
    print("\n" + "-" * 30)
    print(f"模型算法: {model_info.get('model_type', '未知')}")
    print(f"训练日期: {model_info.get('training_date', '未知')}")
    print(f"特征总数: {len(model_info.get('feature_names', []))}")
    print("-" * 30)


if __name__ == "__main__":
    main()