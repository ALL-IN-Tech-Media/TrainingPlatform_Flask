import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

# 1. 创建示例数据
data = {
    "id": range(1, 21),  # 创建20条数据
    "name": [f"用户_{i}" for i in range(1, 21)],
    "age": [20 + i % 5 for i in range(20)],  # 年龄在20-24岁之间
    "score": [round(60 + i * 1.5, 1) for i in range(20)],  # 分数60-90分
    "join_date": [datetime(2023, 1, i % 20 + 1) for i in range(20)],  # 加入日期
    "is_active": [i % 2 == 0 for i in range(20)]  # 是否活跃
}

# # 2. 转换为Pandas DataFrame
# df = pd.DataFrame(data)
# print("原始数据:")
# print(df.head(7))  # 打印前3行看看数据结构

# # 3. 将DataFrame写入Parquet文件
parquet_file = "./Apache Parquet/user_data.parquet"
# df.to_parquet(parquet_file, engine='pyarrow')
# print(f"\n已写入Parquet文件: {parquet_file}")

# # 4. 从Parquet文件读取数据(全部)
# print("\n从Parquet读取全部数据:")
# table = pq.read_table(parquet_file)
# print(table.slice(0, 3).to_pandas())  # 打印前3行

# 5. 高效检索中间的5条数据(第8-12条)
print("\n检索中间的5条数据(第8-12条):")
start_row = 7  # 从0开始计数
num_rows = 5

# 方法1: 读取整个文件后切片(适合小文件)
full_table = pq.read_table(parquet_file)
middle_rows = full_table.slice(start_row, num_rows).to_pandas()
print("方法1结果:")
print(middle_rows)

# # 方法2: 使用行组过滤(适合大文件)
# middle_rows = pq.read_table(
#     parquet_file,
#     use_threads=True,
#     columns=["id", "name", "age"],  # 可选:只选择需要的列
# ).slice(start_row, num_rows).to_pandas()
# print("\n方法2结果(只选择部分列):")
# print(middle_rows)

# # 方法3: 使用谓词下推(过滤条件)
# print("\n检索年龄大于22岁的记录:")
# age_filter = pq.read_table(
#     parquet_file,
#     filters=[("age", ">", 22)]  # 在读取时过滤
# ).to_pandas()
# print(age_filter)