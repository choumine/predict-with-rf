import joblib
import pandas as pd
import os

from fastmcp import FastMCP

mcp = FastMCP("NPS Predict Server")

@mcp.tool
def predict_nps_with_rf(preDefectCount,
                        preClosedDefectCount,
                        preResolvedDefectCount,
                        preTrialDefectCount,
                        preClosedTrialDefectCount,
                        preResolvedTrialDefectCount):
    """
    使用用户调研模型基于6个缺陷变量预测NPS打分
    
    参数:
    preDefectCount: 上市前缺陷数
    preClosedDefectCount: 上市前已关闭缺陷数
    preResolvedDefectCount: 上市前已解决缺陷数
    preTrialDefectCount: 上市前试用缺陷数
    preClosedTrialDefectCount: 上市前已关闭试用缺陷数
    preResolvedTrialDefectCount: 上市前已解决试用缺陷数
    
    返回:
    预测用户调研NPS打分
    """
    
    # 检查模型目录是否存在
    if not os.path.exists("models"):
        raise FileNotFoundError("找不到models目录，请先运行训练脚本")
    
    try:
        # 加载特征名称
        feature_names = joblib.load("models/feature_names.joblib")
        
        # 加载标准化器
        scaler = joblib.load("models/scaler.joblib")
        
        # 加载Random Forest模型
        rf_model = joblib.load("models/random_forest.joblib")
        
    except Exception as e:
        raise Exception(f"加载模型时出错: {e}")
    
    # 创建输入数据字典
    input_data = {
        '上市前缺陷数': preDefectCount,
        '上市前已关闭缺陷数': preClosedDefectCount,
        '上市前已解决缺陷数': preResolvedDefectCount,
        '上市前试用缺陷数': preTrialDefectCount,
        '上市前已关闭试用缺陷数': preClosedTrialDefectCount,
        '上市前已解决试用缺陷数': preResolvedTrialDefectCount
    }
    
    # 转换输入数据为DataFrame
    df = pd.DataFrame([input_data])
    
    # 确保特征顺序正确
    df = df[feature_names]
    
    # 标准化特征
    scaled_data = scaler.transform(df)
    
    # 使用Random Forest模型进行预测
    prediction = rf_model.predict(scaled_data)[0]
    
    return prediction

@mcp.tool
def predict_nps_with_skynet(preDefectCount,
                            preClosedDefectCount,
                            preResolvedDefectCount,
                            preTrialDefectCount,
                            preClosedTrialDefectCount,
                            preResolvedTrialDefectCount):
    """
    使用天网模型基于6个缺陷变量预测NPS打分
    
    参数:
    preDefectCount: 上市前缺陷数
    preClosedDefectCount: 上市前已关闭缺陷数
    preResolvedDefectCount: 上市前已解决缺陷数
    preTrialDefectCount: 上市前试用缺陷数
    preClosedTrialDefectCount: 上市前已关闭试用缺陷数
    preResolvedTrialDefectCount: 上市前已解决试用缺陷数
    
    返回:
    预测天网NPS净推荐值
    """
    
    # 检查模型目录是否存在
    if not os.path.exists("skynet_model"):
        raise FileNotFoundError("找不到skynet_model目录，请先运行训练脚本")
    
    try:
        # 加载特征名称
        feature_names = joblib.load("skynet_model/feature_names.joblib")
        
        # 加载标准化器
        scaler = joblib.load("skynet_model/scaler.joblib")
        
        # 加载Skynet模型
        skynet_model = joblib.load("skynet_model/gradient_boosting.joblib")
        
    except Exception as e:
        raise Exception(f"加载模型时出错: {e}")
    
    # 创建输入数据字典
    input_data = {
        '上市前缺陷数': preDefectCount,
        '上市前已关闭缺陷数': preClosedDefectCount,
        '上市前已解决缺陷数': preResolvedDefectCount,
        '上市前试用缺陷数': preTrialDefectCount,
        '上市前已关闭试用缺陷数': preClosedTrialDefectCount,
        '上市前已解决试用缺陷数': preResolvedTrialDefectCount
    }
    
    # 转换输入数据为DataFrame
    df = pd.DataFrame([input_data])
    
    # 确保特征顺序正确
    df = df[feature_names]
    
    # 标准化特征
    scaled_data = scaler.transform(df)
    
    # 使用Skynet模型进行预测
    prediction = skynet_model.predict(scaled_data)[0]
    
    return prediction


def main():
    """主函数，接收用户输入并预测NPS打分"""
    
    print("=" * 50)
    print("NPS 打分预测工具 (Random Forest版)")
    print("=" * 50)
    
    try:
        # 获取用户输入的特征数据
        print("\n请输入缺陷指标:")
        preDefectCount = float(input("上市前缺陷数: "))
        preClosedDefectCount = float(input("上市前已关闭缺陷数: "))
        preResolvedDefectCount = float(input("上市前已解决缺陷数: "))
        preTrialDefectCount = float(input("上市前试用缺陷数: "))
        preClosedTrialDefectCount = float(input("上市前已关闭试用缺陷数: "))
        preResolvedTrialDefectCount = float(input("上市前已解决试用缺陷数: "))
        
        # 预测
        prediction = predict_nps_with_rf(preDefectCount,
                                         preClosedDefectCount,
                                         preResolvedDefectCount,
                                         preTrialDefectCount,
                                         preClosedTrialDefectCount,
                                         preResolvedTrialDefectCount)
        
        # 显示结果
        print("\n" + "=" * 50)
        print(f"使用 Random Forest 模型预测的 NPS 打分: {prediction:.2f}")
        print("=" * 50)
        
    except ValueError:
        print("请输入有效的数字")
    except Exception as e:
        print(f"预测过程中出错: {e}")


if __name__ == "__main__":
    main()
