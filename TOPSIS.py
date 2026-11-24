"""
TOPSIS
Step0:准备工作
Step1:正向化
Step2:标准化
Step3:计算得分并归一化
"""

# Step0：Preparation
import sys
import os
import pandas as pd
import numpy as np

# 读取表格数据
def ask_filename():
    raw=input(f"Enter file name:")
    filename=raw.strip()
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Input file not found: {filename}")
    print(f"loading data from {filename}\n")
    return filename

# 为主程序做一些准备
# 询问指标类型
def ask_type(name=""):
    while True:    # 输错就重问
        raw=input(
            f"Type of '{name}'? "
            f"[1]max  [2]min  [3]middle  [4]interval（请输入1、2、3、4）: "
        )
        raw=raw.strip()
        if raw in ("1","2","3","4"):
            t = {"1":"max","2":"min","3":"middle","4":"interval"}[raw]
            print(f"'{name}' set to: {t}\n")
            return t
        print("Invalid input, please enter 1 / 2 / 3 / 4.\n")

# 最佳中间值询问
def ask_middle(name=""):
    raw=input(f"Enter the optimal value for '{name}' (e.g.7): ")  # 中文: 请输入最佳值
    best=float(raw.strip())
    print(f"'{name}'optimal value set to {best}\n")
    return best

# 最佳询问区间
def ask_interval(name):
    raw=input("Enter optimal interval [a, b] (e.g.10,20): ")   # 输入最佳区间
    a,b=[float(v)for v in raw.split(",")]
    if a>b:
        raise ValueError(f"Invalid interval: a={a} must be <= b={b}")
    print(f"Optimal interval set to [{a}, {b}]\n")
    return a,b


# Step1:正向化
def min_to_max(x):
    return x.max()-x

def middle_to_max(x,best):
    M=np.max(np.abs(x-best))
    if M==0:
        return np.ones_like(x,dtype=float)
    return 1-np.abs(x-best)/M

def interval_to_max(x,a,b):
    M=max(a-x.min(),x.max()-b)
    result=np.empty_like(x,dtype=float)
    low=x<a
    high=x>b
    mid=~(low|high)
    result[low]=1-(a-x[low])/M
    result[mid]=1.0
    result[high]=1-(x[high]-b)/M
    return result

# Step2:标准化
def normalize(matrix):
    norms=np.sqrt(np.sum(np.square(matrix),axis=0))
    return matrix/norms

# Step3：计算得分并归一化
def topsis(Z,weights=None):
    n,m=Z.shape
    if weights is None:
        weights=np.ones(m)/m   # equal wights
    weights=np.asarray(weights,dtype=float)
    assert abs(weights.sum()-1.0) < 1e-9,"Weights must sum to 1."  # check

    Z_max=Z.max(axis=0)  # Z+
    Z_min=Z.min(axis=0)  # Z-
    D_max=np.sqrt(np.sum(weights*(Z_max-Z)**2,axis=1))  # D+
    D_min=np.sqrt(np.sum(weights*(Z_min-Z)**2,axis=1))  # D-

    S=D_min/(D_max+D_min)
    return S,D_max,D_min

# 主程序(读取数据，Step1、2、3)
if __name__=="__main__":
    FILE=ask_filename()
    df=pd.read_excel(FILE,sheet_name="Sheet1")  # 需要名为“Sheet1”的表格

    objects=df.iloc[:,0].astype(str).tolist()  # 读取第一列对象
    X=df.iloc[:,1:].to_numpy(dtype=float)  # 读取纯数字
    indicator_names=df.columns[1:].tolist()  # 评价指标
    n,m=X.shape

    print("Original decision matrix:")
    print(pd.DataFrame(X,index=objects,columns=indicator_names))

    # 进入正题
    # Step1:正向化
    Xp=np.empty_like(X, dtype=float)
    for j in range(m):  # 逐列处理
        col=X[:,j]
        name=indicator_names[j]

        t=ask_type(name)  # 这列是什么类型

        if t=="max":
            Xp[:,j]=col
        elif t=="min":
            Xp[:,j]=min_to_max(col)
        elif t=="middle":
            best=ask_middle(name)
            Xp[:,j]=middle_to_max(col, best)
        elif t == "interval":
            a,b =ask_interval(name)
            Xp[:,j]=interval_to_max(col,a,b)

    print("\nAfter Step1 (all max-type):")
    print(pd.DataFrame(Xp,index=objects,columns=indicator_names))

    # Step2:标准化
    Z=normalize(Xp)
    print("\nAfter Step2 (normalized matrix Z):")
    print(pd.DataFrame(Z,index=objects,columns=indicator_names))

    # Step3:计算得分并排序
    S,D_max,D_min=topsis(Z)  # 若已有权重, 改为:S,D_max,D_min=topsis(Z, weights=w_ahp)

    result = pd.DataFrame({
        "D+ (to ideal)": D_max,
        "D- (to worst)": D_min,
        "Score S":S,
    }, index=objects)
    print("\n",result)

    order=np.argsort(-S)
    print("\nFinal ranking:")
    for rank,idx in enumerate(order, start=1):
        print(f"TOP{rank}:{objects[idx]}={S[idx]:.4f}")

    # 导出结果
    result.to_csv("topsis_results.csv", encoding="utf-8-sig")
    print("\nSaved: topsis_results.csv")