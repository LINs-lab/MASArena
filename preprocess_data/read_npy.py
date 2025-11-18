import numpy as np
import pandas as pd

# 设置numpy打印选项
np.set_printoptions(suppress=True, threshold=10)

# 定义文件路径
embeddings_file = 'preprocess_data/knowledge_base_plan_embeddings.npy'
similarity_file = 'preprocess_data/knowledge_base_plan_similarity.npy'

# 读取相似度矩阵并保存到Excel
print(f"--- Loading similarity matrix from: {similarity_file} ---")
try:
    similarity_matrix = np.load(similarity_file)
    print("Shape of similarity matrix:", similarity_matrix.shape)
    print("Data type:", similarity_matrix.dtype)
    
    # 计算矩阵总元素个数
    total_elements = similarity_matrix.size
    print(f"矩阵总元素个数: {total_elements}")
    
    # 计算大于0.5的元素个数
    elements_above_threshold = np.sum(similarity_matrix > 0.4)
    print(f"大于0.5的元素个数: {elements_above_threshold}")
    
    # 计算占比
    ratio = elements_above_threshold / total_elements
    print(f"大于0.5的元素占比: {ratio:.4f} ({ratio*100:.2f}%)")
    
    # # 1. 保存完整矩阵到Excel
    # df_full = pd.DataFrame(similarity_matrix)
    # df_full.to_excel('plan_similarity_matrix_full.xlsx', index=False, header=False)
    # print("完整相似度矩阵已保存到 'plan_similarity_matrix_full.xlsx'")
    
    # # 2. 创建过滤后的矩阵（只保留大于0.3的值，其余设为NaN）
    # filtered_matrix = similarity_matrix.copy()
    # filtered_matrix[filtered_matrix <= 0.3] = np.nan
    
    # 保存过滤后的矩阵
    # df_filtered = pd.DataFrame(filtered_matrix)
    # df_filtered.to_excel('plan_similarity_matrix_filtered.xlsx', index=False, header=False)
    # print("过滤后的相似度矩阵已保存到 'plan_similarity_matrix_filtered.xlsx'")
    
    # # 显示完整矩阵的预览
    # print(f"\n完整相似度矩阵预览 (前10x10):")
    # print(df_full.iloc[:10, :10])
    
    # 显示过滤后矩阵的预览
    # print(f"\n过滤后矩阵预览 (前10x10):")
    # print(df_filtered.iloc[:10, :10])
    
    # 添加统计信息总结
    print(f"\n=== 统计信息总结 ===")
    print(f"矩阵维度: {similarity_matrix.shape}")
    print(f"总元素数: {total_elements}")
    print(f"相似度 > 0.5 的元素数: {elements_above_threshold}")
    print(f"占比: {ratio:.4f} ({ratio*100:.2f}%)")
    print(f"相似度 <= 0.5 的元素数: {total_elements - elements_above_threshold}")
    
except FileNotFoundError:
    print(f"Error: File not found at {similarity_file}")
except Exception as e:
    print(f"An error occurred: {e}")